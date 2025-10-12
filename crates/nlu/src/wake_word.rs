// crates/nlu/src/wake_word.rs

/// Нечёткое сравнение wake words с поддержкой опечаток
pub fn fuzzy_wake_word_match(text: &str, wake_words: &[String]) -> Option<(String, usize)> {
    let text_lower = text.to_lowercase();

    for wake_word in wake_words {
        let wake_lower = wake_word.to_lowercase();

        // 1. Точное совпадение
        if text_lower.contains(&wake_lower) {
            return Some((wake_word.clone(), 0));
        }

        // 2. Фонетические варианты (русский)
        let phonetic_variants = get_phonetic_variants(&wake_lower);
        for variant in phonetic_variants {
            if text_lower.contains(&variant) {
                return Some((wake_word.clone(), 0));
            }
        }

        // 3. Levenshtein distance для опечаток
        for word in text_lower.split_whitespace() {
            let distance = levenshtein_distance(word, &wake_lower);
            let max_allowed = (wake_lower.len() / 4).max(1); // 25% ошибок допустимо

            if distance <= max_allowed {
                return Some((wake_word.clone(), distance));
            }
        }
    }

    None
}

/// Фонетические варианты для русских слов
fn get_phonetic_variants(word: &str) -> Vec<String> {
    let mut variants = vec![];

    // Общие замены букв в русском языке при распознавании
    let replacements = vec![
        ("к", "г"), ("г", "к"),     // Кларнет -> Гларнет
        ("т", "д"), ("д", "т"),     // Кларнет -> Кларнед
        ("е", "и"), ("и", "е"),     // Кларнет -> Кларнит
        ("о", "а"), ("а", "о"),
        ("ё", "е"), ("е", "ё"),
        ("й", "и"), ("и", "й"),
    ];

    for (from, to) in replacements {
        if word.contains(from) {
            variants.push(word.replace(from, to));
        }
    }

    variants
}

/// Вычисление расстояния Левенштейна
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 { return b_len; }
    if b_len == 0 { return a_len; }

    let mut matrix = vec![vec![0; b_len + 1]; a_len + 1];

    for i in 0..=a_len { matrix[i][0] = i; }
    for j in 0..=b_len { matrix[0][j] = j; }

    for i in 1..=a_len {
        for j in 1..=b_len {
            let cost = if a_chars[i-1] == b_chars[j-1] { 0 } else { 1 };
            matrix[i][j] = (matrix[i-1][j] + 1)
                .min(matrix[i][j-1] + 1)
                .min(matrix[i-1][j-1] + cost);
        }
    }

    matrix[a_len][b_len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let wake_words = vec!["Кларнет".to_string()];
        let result = fuzzy_wake_word_match("Кларнет включи свет", &wake_words);
        assert!(result.is_some());
        assert_eq!(result.unwrap().1, 0); // distance = 0
    }

    #[test]
    fn test_phonetic_match() {
        let wake_words = vec!["Кларнет".to_string()];
        // "Гларнет" - фонетически похоже
        let result = fuzzy_wake_word_match("Гларнет какая погода", &wake_words);
        assert!(result.is_some());
    }

    #[test]
    fn test_typo_match() {
        let wake_words = vec!["Кларнет".to_string()];
        // "Ларнет" - пропущена одна буква
        let result = fuzzy_wake_word_match("Ларнет какая погода", &wake_words);
        assert!(result.is_some());
        assert!(result.unwrap().1 <= 2); // небольшое расстояние
    }

    #[test]
    fn test_no_match() {
        let wake_words = vec!["Кларнет".to_string()];
        let result = fuzzy_wake_word_match("Привет мир", &wake_words);
        assert!(result.is_none());
    }
}