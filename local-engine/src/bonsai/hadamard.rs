use std::collections::HashMap;
use std::sync::Arc;

use local_metal::bonsai::SignedHadamard;
use local_metal::context::MetalContext;

use super::{
    BonsaiTensor, BonsaiTensorType, HadamardDirection, MetadataValue, array_i32, array_strings,
    invalid, scalar_bool, scalar_string, scalar_u32,
};

#[derive(Clone, Debug)]
pub struct HadamardMetadata {
    signs: HashMap<u32, Arc<[f32]>>,
    directions: HashMap<String, HadamardDirection>,
    gdn_v_grouped: bool,
}

impl HadamardMetadata {
    pub fn signs(&self, width: u32) -> crate::Result<&[f32]> {
        self.signs.get(&width).map(AsRef::as_ref).ok_or_else(|| {
            crate::Error::InvalidFormat(format!("no checkpoint signs for width {width}"))
        })
    }
    pub fn direction(&self, name: &str) -> crate::Result<HadamardDirection> {
        self.directions.get(name).copied().ok_or_else(|| {
            crate::Error::InvalidFormat(format!("tensor {name} has no Hadamard direction"))
        })
    }
    #[must_use]
    pub const fn gdn_v_grouped(&self) -> bool {
        self.gdn_v_grouped
    }
    pub(crate) fn all_signs(&self) -> impl Iterator<Item = (u32, &[f32])> {
        self.signs
            .iter()
            .map(|(&width, signs)| (width, signs.as_ref()))
    }
    /// The checkpoint's signs for `width` as a native rotation operand.
    pub fn upload(&self, context: &MetalContext, width: u32) -> crate::Result<SignedHadamard> {
        SignedHadamard::new(context, self.signs(width)?).map_err(crate::Error::Metal)
    }
}

fn forward_weight_name(name: &str) -> bool {
    if name == "output.weight" {
        return true;
    }
    let Some((layer, suffix)) = name
        .strip_prefix("blk.")
        .and_then(|name| name.split_once('.'))
    else {
        return false;
    };
    layer.parse::<u32>().is_ok()
        && matches!(
            suffix,
            "attn_q.weight"
                | "attn_k.weight"
                | "attn_v.weight"
                | "attn_qkv.weight"
                | "attn_gate.weight"
                | "attn_output.weight"
                | "ffn_gate.weight"
                | "ffn_up.weight"
                | "ffn_down.weight"
                | "ssm_out.weight"
        )
}

#[allow(clippy::too_many_lines)]
pub(super) fn parse_hadamard(
    map: &[u8],
    metadata: &HashMap<String, MetadataValue>,
    tensors: &[BonsaiTensor],
    by_name: &HashMap<String, usize>,
) -> crate::Result<HadamardMetadata> {
    let require = |key: &str| {
        metadata
            .get(key)
            .ok_or_else(|| crate::Error::InvalidFormat(format!("missing required metadata {key}")))
    };
    if scalar_u32(map, require("prism.hadamard.version")?)? != 1
        || scalar_u32(map, require("prism.hadamard.block_size")?)? != 1024
        || scalar_string(map, require("prism.hadamard.transform")?)?
            != "normalized-sylvester-walsh-hadamard"
        || scalar_string(map, require("prism.hadamard.axis")?)? != "input-last-dimension"
        || scalar_string(map, require("prism.hadamard.sign_mode")?)? != "explicit"
    {
        return invalid("unsupported prism.hadamard profile");
    }
    let widths = array_i32(map, require("prism.hadamard.sign_widths")?)?;
    let values = array_i32(map, require("prism.hadamard.sign_values")?)?;
    let mut signs = HashMap::new();
    let mut at = 0_usize;
    for width in widths {
        let width = u32::try_from(width)
            .map_err(|_| crate::Error::InvalidFormat("invalid Hadamard sign width".into()))?;
        if width == 0 || !width.is_multiple_of(1024) || signs.contains_key(&width) {
            return invalid("invalid or duplicate Hadamard sign width");
        }
        let end = at
            .checked_add(width as usize)
            .ok_or_else(|| crate::Error::InvalidFormat("sign count overflow".into()))?;
        let slice = values
            .get(at..end)
            .ok_or_else(|| crate::Error::InvalidFormat("missing Hadamard signs".into()))?;
        if slice.iter().any(|&v| v != -1 && v != 1) {
            return invalid("Hadamard signs must be +/-1");
        }
        signs.insert(
            width,
            slice.iter().map(|&v| v as f32).collect::<Vec<_>>().into(),
        );
        at = end;
    }
    if at != values.len() {
        return invalid("Hadamard sign totals do not match widths");
    }
    let forward = array_strings(map, require("prism.hadamard.weight_names")?)?;
    let inverse = array_strings(map, require("prism.hadamard.inverse_weight_names")?)?;
    if forward.is_empty()
        || forward.iter().any(|name| !forward_weight_name(name))
        || inverse != ["token_embd.weight"]
    {
        return invalid("unsupported Bonsai forward/inverse rotation targets");
    }
    let mut directions = HashMap::new();
    for (names, direction) in [
        (forward, HadamardDirection::Forward),
        (inverse, HadamardDirection::Inverse),
    ] {
        for name in names {
            let tensor = by_name.get(&name).map(|&i| &tensors[i]).ok_or_else(|| {
                crate::Error::InvalidFormat(format!("Hadamard names missing tensor {name}"))
            })?;
            if tensor.tensor_type != BonsaiTensorType::Ptq1
                || tensor.dimensions.len() != 2
                || u32::try_from(tensor.dimensions[0]).is_err()
                || u32::try_from(tensor.dimensions[1]).is_err()
                || !signs.contains_key(&(tensor.dimensions[0] as u32))
            {
                return invalid(format!("Hadamard width/type mismatch for {name}"));
            }
            if directions.insert(name.clone(), direction).is_some() {
                return invalid(format!("duplicate/conflicting Hadamard tensor {name}"));
            }
        }
    }
    for tensor in tensors
        .iter()
        .filter(|t| t.tensor_type == BonsaiTensorType::Ptq1)
    {
        if !directions.contains_key(&tensor.name) {
            return invalid(format!(
                "PTQ1 tensor {} lacks Hadamard coverage",
                tensor.name
            ));
        }
    }
    let grouped = scalar_bool(map, require("prism.hadamard.gdn_v_grouped")?)?;
    if !grouped {
        return invalid("pinned Bonsai profile requires grouped GDN-V contract");
    }
    Ok(HadamardMetadata {
        signs,
        directions,
        gdn_v_grouped: grouped,
    })
}
