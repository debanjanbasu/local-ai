use super::{MAX_HEADER_BYTES, MetadataValue, invalid};

pub(super) struct Cursor<'a> {
    bytes: &'a [u8],
    pub(super) pos: usize,
    limit: usize,
}
impl<'a> Cursor<'a> {
    pub(super) fn new(bytes: &'a [u8], limit: usize) -> Self {
        Self {
            bytes,
            pos: 0,
            limit: bytes.len().min(limit),
        }
    }
    pub(super) fn take(&mut self, count: usize) -> crate::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(count)
            .ok_or_else(|| crate::Error::InvalidFormat("GGUF cursor overflow".into()))?;
        if end > self.limit {
            return invalid("truncated or oversized GGUF header");
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }
    pub(super) fn u32(&mut self) -> crate::Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    pub(super) fn u64(&mut self) -> crate::Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    pub(super) fn string(&mut self) -> crate::Result<&'a str> {
        let n = usize::try_from(self.u64()?)
            .map_err(|_| crate::Error::InvalidFormat("string exceeds address space".into()))?;
        std::str::from_utf8(self.take(n)?)
            .map_err(|_| crate::Error::InvalidFormat("GGUF string is not UTF-8".into()))
    }
}
fn skip_scalars(cursor: &mut Cursor<'_>, ty: u32, count: usize) -> crate::Result<()> {
    let width = match ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        8 => {
            for _ in 0..count {
                cursor.string()?;
            }
            return Ok(());
        }
        9 => return invalid("nested GGUF arrays are unsupported"),
        _ => return invalid(format!("unknown GGUF metadata type {ty}")),
    };
    let length = count
        .checked_mul(width)
        .ok_or_else(|| crate::Error::InvalidFormat("metadata array byte length overflow".into()))?;
    let data = cursor.take(length)?;
    if ty == 7 && data.iter().any(|&value| value > 1) {
        return invalid("invalid GGUF bool");
    }
    Ok(())
}

pub(super) fn skip_value(
    cursor: &mut Cursor<'_>,
    ty: u32,
) -> crate::Result<(Option<u32>, Option<u64>)> {
    if ty != 9 {
        skip_scalars(cursor, ty, 1)?;
        return Ok((None, None));
    }
    let element = cursor.u32()?;
    let count = cursor.u64()?;
    if count > MAX_HEADER_BYTES as u64 {
        return invalid("GGUF array count exceeds bound");
    }
    skip_scalars(cursor, element, count as usize)?;
    Ok((Some(element), Some(count)))
}
pub(super) fn value_cursor<'a>(map: &'a [u8], v: &MetadataValue) -> Cursor<'a> {
    let mut c = Cursor::new(map, v.payload.end);
    c.pos = v.payload.start;
    c
}
pub(super) fn scalar_u32(map: &[u8], v: &MetadataValue) -> crate::Result<u32> {
    if v.ty != 4 {
        return invalid("metadata type mismatch: expected u32");
    }
    value_cursor(map, v).u32()
}
pub(super) fn scalar_bool(map: &[u8], v: &MetadataValue) -> crate::Result<bool> {
    if v.ty != 7 {
        return invalid("metadata type mismatch: expected bool");
    }
    match value_cursor(map, v).take(1)?[0] {
        0 => Ok(false),
        1 => Ok(true),
        _ => invalid("invalid GGUF bool"),
    }
}
pub(super) fn scalar_string<'a>(map: &'a [u8], v: &MetadataValue) -> crate::Result<&'a str> {
    if v.ty != 8 {
        return invalid("metadata type mismatch: expected string");
    }
    value_cursor(map, v).string()
}
pub(super) fn array_i32(map: &[u8], v: &MetadataValue) -> crate::Result<Vec<i32>> {
    if v.ty != 9 || v.element_ty != Some(5) {
        return invalid("metadata type mismatch: expected i32 array");
    }
    let mut c = value_cursor(map, v);
    if c.u32()? != 5 {
        return invalid("array element mismatch");
    }
    let count = c.u64()?;
    if Some(count) != v.count {
        return invalid("array count mismatch");
    }
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let b = c.take(4)?;
        out.push(i32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    Ok(out)
}
pub(super) fn array_strings(map: &[u8], v: &MetadataValue) -> crate::Result<Vec<String>> {
    if v.ty != 9 || v.element_ty != Some(8) {
        return invalid("metadata type mismatch: expected string array");
    }
    let mut c = value_cursor(map, v);
    if c.u32()? != 8 {
        return invalid("array element mismatch");
    }
    let count = c.u64()?;
    if Some(count) != v.count {
        return invalid("array count mismatch");
    }
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        out.push(c.string()?.to_owned());
    }
    Ok(out)
}
