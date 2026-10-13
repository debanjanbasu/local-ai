//! English number words in `num2words`'s forms (cardinal, ordinal, year,
//! decimal), which `misaki` spells numbers with. Only the word sequence
//! matters to the G2P (it drops "and" and punctuation), so "and" placement
//! follows `num2words` (British style) but commas are omitted.

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [(u64, &str); 6] = [
    (1_000_000_000_000_000_000, "quintillion"),
    (1_000_000_000_000_000, "quadrillion"),
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
];

fn below_thousand(n: u64) -> String {
    let (hundreds, rest) = (n / 100, n % 100);
    let tail = if rest < 20 {
        ONES[rest as usize].to_owned()
    } else if rest % 10 == 0 {
        TENS[(rest / 10) as usize].to_owned()
    } else {
        format!("{}-{}", TENS[(rest / 10) as usize], ONES[(rest % 10) as usize])
    };
    match (hundreds, rest) {
        (0, _) => tail,
        (h, 0) => format!("{} hundred", ONES[h as usize]),
        (h, _) => format!("{} hundred and {tail}", ONES[h as usize]),
    }
}

/// Cardinal words: 123 -> "one hundred and twenty-three".
#[must_use]
pub fn cardinal(n: u64) -> String {
    if n < 1000 {
        return below_thousand(n);
    }
    let mut parts = Vec::new();
    let mut rest = n;
    for (scale, name) in SCALES {
        if rest >= scale {
            parts.push(format!("{} {name}", cardinal(rest / scale)));
            rest %= scale;
        }
    }
    if rest > 0 {
        if rest < 100 {
            parts.push(format!("and {}", below_thousand(rest)));
        } else {
            parts.push(below_thousand(rest));
        }
    }
    parts.join(" ")
}

/// Cardinal words of a decimal digit string (saturating on overflow).
#[must_use]
pub fn cardinal_str(digits: &str) -> String {
    digits.parse::<u64>().map_or_else(
        |_| digits.chars().filter_map(|c| c.to_digit(10)).map(|d| ONES[d as usize]).collect::<Vec<_>>().join(" "),
        cardinal,
    )
}

/// Ordinal words: 14 -> "fourteenth".
#[must_use]
pub fn ordinal_str(digits: &str) -> String {
    let words = cardinal_str(digits);
    let (head, last) = match words.rfind([' ', '-']) {
        Some(at) => (&words[..=at], &words[at + 1..]),
        None => ("", words.as_str()),
    };
    let last = match last {
        "one" => "first".to_owned(),
        "two" => "second".to_owned(),
        "three" => "third".to_owned(),
        "five" => "fifth".to_owned(),
        "eight" => "eighth".to_owned(),
        "nine" => "ninth".to_owned(),
        "twelve" => "twelfth".to_owned(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    };
    format!("{head}{last}")
}

/// Year words: 1999 -> "nineteen ninety-nine", 2005 -> "two thousand and
/// five", 1905 -> "nineteen oh-five", 1900 -> "nineteen hundred".
#[must_use]
pub fn year_str(digits: &str) -> String {
    let Ok(value) = digits.parse::<u64>() else {
        return cardinal_str(digits);
    };
    let (high, low) = (value / 100, value % 100);
    if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
        return cardinal(value);
    }
    let low = match low {
        0 => "hundred".to_owned(),
        1..=9 => format!("oh-{}", cardinal(low)),
        _ => cardinal(low),
    };
    format!("{} {low}", cardinal(high))
}

/// Decimal words: "3.50" -> "three point five" (`num2words(float)` drops
/// trailing zeros of the float's shortest repr).
#[must_use]
pub fn float_str(text: &str) -> String {
    let Ok(value) = text.parse::<f64>() else {
        return cardinal_str(text);
    };
    let repr = format!("{value:?}");
    let (int, frac) = repr.split_once('.').unwrap_or((&repr, "0"));
    let int = cardinal_str(int.trim_start_matches('-'));
    let digits: Vec<&str> = frac
        .chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| ONES[d as usize])
        .collect();
    format!("{int} point {}", digits.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_follow_num2words() {
        assert_eq!(cardinal(0), "zero");
        assert_eq!(cardinal(21), "twenty-one");
        assert_eq!(cardinal(123), "one hundred and twenty-three");
        assert_eq!(cardinal(1005), "one thousand and five");
        assert_eq!(cardinal(2_300_000), "two million three hundred thousand");
        assert_eq!(ordinal_str("14"), "fourteenth");
        assert_eq!(ordinal_str("21"), "twenty-first");
        assert_eq!(ordinal_str("40"), "fortieth");
        assert_eq!(year_str("1999"), "nineteen ninety-nine");
        assert_eq!(year_str("2005"), "two thousand and five");
        assert_eq!(year_str("1905"), "nineteen oh-five");
        assert_eq!(year_str("2010"), "twenty ten");
        assert_eq!(float_str("3.50"), "three point five");
    }
}
