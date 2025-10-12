use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const COMMON_REPLACEMENTS: &[(&str, &str)] = &[
    // Latin transliterations that Whisper may emit
    ("shch", "щ"),
    ("sch", "щ"),
    ("sh", "ш"),
    ("ch", "ч"),
    ("yo", "ё"),
    ("ye", "е"),
    ("ya", "я"),
    ("yu", "ю"),
    ("ja", "я"),
    ("ju", "ю"),
    ("je", "е"),
    ("jo", "ё"),
    ("ph", "ф"),
    ("ts", "ц"),
    ("kh", "х"),
    ("qu", "к"),
    ("ck", "к"),
    // Cyrillic assimilations and frequent recognition glitches
    ("дж", "ж"),
    ("дз", "з"),
    ("сч", "щ"),
    ("зч", "жч"),
    ("сш", "ш"),
    ("зж", "ж"),
    ("стн", "сн"),
    ("здн", "зн"),
    ("тс", "ц"),
    ("ться", "ца"),
    ("тся", "ца"),
];

/// Lightweight phonetic encoder for Russian commands.
#[derive(Debug, Clone)]
pub struct RussianPhonetic {
    char_map: HashMap<char, char>,
}

impl Default for RussianPhonetic {
    fn default() -> Self {
        Self::new()
    }
}

impl RussianPhonetic {
    pub fn new() -> Self {
        let mut char_map = HashMap::new();

        for c in ['к', 'г', 'х'] {
            char_map.insert(c, '1');
        }
        for c in ['ц', 'с', 'з'] {
            char_map.insert(c, '2');
        }
        for c in ['т', 'д'] {
            char_map.insert(c, '3');
        }
        for c in ['б', 'п'] {
            char_map.insert(c, '4');
        }
        for c in ['ф', 'в'] {
            char_map.insert(c, '5');
        }
        for c in ['м', 'н'] {
            char_map.insert(c, '6');
        }
        for c in ['л', 'р'] {
            char_map.insert(c, '7');
        }
        for c in ['ш', 'ж', 'щ', 'ч'] {
            char_map.insert(c, '8');
        }
        char_map.insert('й', '9');

        for c in ['а', 'о', 'у', 'ы', 'э'] {
            char_map.insert(c, 'A');
        }
        for c in ['и', 'е', 'ё', 'ю', 'я'] {
            char_map.insert(c, 'I');
        }

        // Latin characters that Whisper can emit when confident in phonetics
        for c in ['k', 'g', 'h', 'q', 'x'] {
            char_map.insert(c, '1');
        }
        for c in ['c', 's', 'z'] {
            char_map.insert(c, '2');
        }
        for c in ['t', 'd'] {
            char_map.insert(c, '3');
        }
        for c in ['b', 'p'] {
            char_map.insert(c, '4');
        }
        for c in ['f', 'v', 'w'] {
            char_map.insert(c, '5');
        }
        for c in ['m', 'n'] {
            char_map.insert(c, '6');
        }
        for c in ['l', 'r'] {
            char_map.insert(c, '7');
        }
        char_map.insert('j', '9');
        char_map.insert('y', '9');

        for c in ['a', 'o', 'u'] {
            char_map.insert(c, 'A');
        }
        for c in ['e', 'i'] {
            char_map.insert(c, 'I');
        }

        for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] {
            char_map.insert(digit, '0');
        }


        Self { char_map }
    }

    pub fn encode(&self, text: &str) -> String {
        let tokens = self.split_tokens(text);
        let mut encoded = Vec::new();

        for token in tokens {
            if let Some(normalized) = self.preprocess_token(&token) {
                let code = self.encode_preprocessed(&normalized);
                if !code.is_empty() {
                    encoded.push(code);
                }
            }
        }

        encoded.join("-")
    }

    fn split_tokens(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() || (ch as u32) >= 0x0400 && (ch as u32) <= 0x04FF {
                for lower in ch.to_lowercase() {
                    current.push(lower);
                }
            } else if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    fn preprocess_token(&self, token: &str) -> Option<String> {
        let mut normalized = String::with_capacity(token.len());

        for ch in token.chars() {
            match ch {
                'ь' | 'ъ' => continue,
                'ё' => normalized.push('е'),
                other if other.is_alphabetic() || other.is_ascii_digit() => {
                    normalized.push(other);
                }
                _ => {}
            }
        }

        if normalized.is_empty() {
            return None;
        }

        let mut replaced = normalized;
        for (pattern, replacement) in COMMON_REPLACEMENTS {
            if replaced.contains(pattern) {
                replaced = replaced.replace(pattern, replacement);
            }
        }

        Some(replaced)
    }

    fn encode_preprocessed(&self, token: &str) -> String {
        let mut code = String::new();
        let mut last_code = '\0';

        for ch in token.chars() {
            if let Some(&phoneme) = self.char_map.get(&ch) {
                if phoneme != last_code {
                    code.push(phoneme);
                    last_code = phoneme;
                }
            } else if ch.is_whitespace() {
                if !code.is_empty() && !code.ends_with('-') {
                    code.push('-');
                    last_code = '\0';
                }
            }
        }

        code
    }

    pub fn encode_token(&self, token: &str) -> String {
        match self.preprocess_token(token) {
            Some(preprocessed) => self.encode_preprocessed(&preprocessed),
            None => String::new(),
        }
    }

    pub fn distance(&self, code1: &str, code2: &str) -> usize {
        let len1 = code1.len();
        let len2 = code2.len();
        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let chars1: Vec<char> = code1.chars().collect();
        let chars2: Vec<char> = code2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    pub fn is_similar(&self, text1: &str, text2: &str, max_distance: usize) -> bool {
        let code1 = self.encode(text1);
        let code2 = self.encode(text2);
        self.distance(&code1, &code2) <= max_distance
    }
}

/// Dictionary that keeps phonetic encodings for seen tokens.
#[derive(Debug, Clone)]
pub struct PhoneticDictionary {
    encoder: RussianPhonetic,
    words: HashMap<String, String>,
}

impl Default for PhoneticDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticDictionary {
    pub fn new() -> Self {
        Self {
            encoder: RussianPhonetic::new(),
            words: HashMap::new(),
        }
    }

    pub fn register_text(&mut self, text: &str) -> String {
        let mut codes = Vec::new();

        for token in self.encoder.split_tokens(text) {
            if let Some(normalized) = self.encoder.preprocess_token(&token) {
                let code = self
                    .words
                    .entry(normalized.clone())
                    .or_insert_with(|| self.encoder.encode_preprocessed(&normalized))
                    .clone();

                if !code.is_empty() {
                    codes.push(code);
                }
            }
        }

        codes.join("-")
    }

    pub fn encode(&self, text: &str) -> String {
        let mut codes = Vec::new();

        for token in self.encoder.split_tokens(text) {
            if let Some(normalized) = self.encoder.preprocess_token(&token) {
                if let Some(code) = self.words.get(&normalized) {
                    if !code.is_empty() {
                        codes.push(code.clone());
                    }
                } else {
                    let generated = self.encoder.encode_preprocessed(&normalized);
                    if !generated.is_empty() {
                        codes.push(generated);
                    }
                }
            }
        }

        codes.join("-")
    }

    pub fn distance(&self, left: &str, right: &str) -> usize {
        self.encoder.distance(left, right)
    }

    pub fn encoder(&self) -> &RussianPhonetic {
        &self.encoder
    }
}

/// Phonetic command entry stored in the matcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneticCommand {
    pub text: String,
    pub code: String,
    pub intent: String,
    pub confidence: f32,
}

/// Result of a phonetic lookup including edit distance for scoring.
#[derive(Debug, Clone)]
pub struct PhoneticMatch {
    pub command: PhoneticCommand,
    pub distance: usize,
    pub input_code: String,
}

pub struct PhoneticMatcher {
    dictionary: PhoneticDictionary,
    commands: Vec<PhoneticCommand>,
}

impl Default for PhoneticMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl PhoneticMatcher {
    pub fn new() -> Self {
        Self {
            dictionary: PhoneticDictionary::new(),
            commands: Vec::new(),
        }
    }

    pub fn add_command(&mut self, text: &str, intent: &str, confidence: f32) {
        let code = self.dictionary.register_text(text);
        self.commands.push(PhoneticCommand {
            text: text.to_string(),
            code,
            intent: intent.to_string(),
            confidence,
        });
    }

    pub fn load_from_patterns(&mut self, patterns: &[(&str, &str)]) {
        for (pattern, intent) in patterns {
            self.add_command(pattern, intent, 0.9);
        }
    }

    pub fn match_text(&self, text: &str) -> Option<PhoneticMatch> {
        self.match_text_within(text, usize::MAX)
    }

    pub fn match_text_within(&self, text: &str, max_distance: usize) -> Option<PhoneticMatch> {
        if self.commands.is_empty() {
            return None;
        }

        let input_code = self.dictionary.encode(text);
        let mut best: Option<(usize, &PhoneticCommand)> = None;

        for command in &self.commands {
            let distance = self.dictionary.distance(&input_code, &command.code);
            if distance > max_distance {
                continue;
            }

            match &mut best {
                Some((best_distance, best_command)) => {
                    if distance < *best_distance
                        || (distance == *best_distance
                            && command.confidence > best_command.confidence)
                    {
                        *best_distance = distance;
                        *best_command = command;
                    }
                }
                None => best = Some((distance, command)),
            }
        }

        best.map(|(distance, command)| PhoneticMatch {
            command: command.clone(),
            distance,
            input_code,
        })
    }

    pub fn needs_llm(&self, text: &str) -> bool {
        let question_words = ["как", "что", "где", "когда", "почему", "зачем", "сколько"];
        let lower = text.to_lowercase();

        if question_words.iter().any(|word| lower.contains(word)) {
            return true;
        }

        lower.split_whitespace().count() > 10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_groups_similar_sounds() {
        let encoder = RussianPhonetic::new();
        assert_eq!(encoder.encode("включи"), encoder.encode("фключи"));
        assert_eq!(encoder.encode("кларнет"), encoder.encode("гларнет"));
    }

    #[test]
    fn matcher_finds_close_matches() {
        let mut matcher = PhoneticMatcher::new();
        matcher.add_command("включи свет", "lights_on", 0.9);
        matcher.add_command("выключи свет", "lights_off", 0.9);

        let result = matcher.match_text_within("фключи сфет", 2).expect("match");
        assert_eq!(result.command.intent, "lights_on");
    }

    #[test]
    fn dictionary_reuses_known_tokens() {
        let mut dictionary = PhoneticDictionary::new();
        let expected = dictionary.register_text("включи свет");

        assert_eq!(expected, dictionary.encode("включи свет"));
        assert_eq!(expected, dictionary.encode("фключи сфет"));
    }

    #[test]
    fn needs_llm_identifies_questions() {
        let matcher = PhoneticMatcher::new();
        assert!(matcher.needs_llm("как погода"));
        assert!(!matcher.needs_llm("включи свет"));
    }
}