// src/main.rs

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

// Максимальна довжина слова, яку ми будемо розглядати.
const MAX_WORD_LEN: usize = 25;
// Константа для згладжування (використовується для слів/пар, яких немає в моделі)
const SMOOTHING: f64 = 1e-10;

// --- Мовні моделі ---
// Робимо структуру публічною, щоб виправити попередження компілятора
pub struct LanguageModel {
    unigram_log_probs: HashMap<String, f64>,
    bigram_log_probs: HashMap<String, HashMap<String, f64>>,
    dictionary_by_len: HashMap<usize, Vec<String>>,
}

impl LanguageModel {
    pub fn new(dict_path: &str, corpus_path: &str) -> Result<Self, std::io::Error> {
        println!("Loading dictionary...");
        let dictionary_by_len = Self::load_dictionary(dict_path)?;
        
        println!("Dictionary loaded. Building language models...");
        let (unigram_log_probs, bigram_log_probs) = Self::build_prob_models(corpus_path)?;
        println!("Language models built.");

        Ok(Self {
            unigram_log_probs,
            bigram_log_probs,
            dictionary_by_len,
        })
    }

    fn load_dictionary(path: &str) -> Result<HashMap<usize, Vec<String>>, std::io::Error> {
        let content = fs::read_to_string(path)?;
        let mut dictionary = HashMap::new();
        for word in content.lines() {
            let word = word.trim().to_lowercase();
            if !word.is_empty() && word.len() <= MAX_WORD_LEN {
                dictionary.entry(word.len()).or_insert_with(Vec::new).push(word);
            }
        }
        Ok(dictionary)
    }

    // Будуємо ймовірнісні моделі (уніграми та біграми)
    fn build_prob_models(path: &str) -> Result<(HashMap<String, f64>, HashMap<String, HashMap<String, f64>>), std::io::Error> {
        let content = fs::read_to_string(path)?.to_lowercase();
        let words: Vec<&str> = content.split(|c: char| !c.is_alphabetic()).filter(|s| !s.is_empty()).collect();
        let total_words = words.len() as f64;

        let mut unigram_freqs = HashMap::new();
        let mut bigram_freqs = HashMap::new();
        let mut prev_word = "<START>".to_string();

        for word in &words {
            *unigram_freqs.entry(word.to_string()).or_insert(0) += 1;
            let followers = bigram_freqs.entry(prev_word.clone()).or_insert_with(HashMap::new);
            *followers.entry(word.to_string()).or_insert(0) += 1;
            prev_word = word.to_string();
        }

        // Перетворюємо частоти в логарифми ймовірностей
        let unigram_log_probs = unigram_freqs
            .into_iter()
            .map(|(word, count)| (word, (count as f64 / total_words).log10()))
            .collect();
        
        let bigram_log_probs = bigram_freqs
            .into_iter()
            .map(|(prev, followers)| {
                let total_followers = followers.values().sum::<u32>() as f64;
                let log_probs = followers
                    .into_iter()
                    .map(|(curr, count)| (curr, (count as f64 / total_followers).log10()))
                    .collect();
                (prev, log_probs)
            })
            .collect();

        Ok((unigram_log_probs, bigram_log_probs))
    }
}

// --- Алгоритм Вітербі ---
pub struct ViterbiDecoder<'a> {
    model: &'a LanguageModel,
}

impl<'a> ViterbiDecoder<'a> {
    pub fn new(model: &'a LanguageModel) -> Self {
        Self { model }
    }

    // Перевірка відповідності слова із "*" та перемішуваннями
    fn is_match(corrupted: &str, word: &str) -> bool {
        if corrupted.len() != word.len() { return false; }
        let mut word_freq = HashMap::new();
        word.chars().for_each(|c| *word_freq.entry(c).or_insert(0) += 1);
        for c in corrupted.chars() {
            if c == '*' { continue; }
            if let Some(count) = word_freq.get_mut(&c) {
                if *count > 0 { *count -= 1; } else { return false; }
            } else { return false; }
        }
        true
    }

    // Отримання логарифму ймовірності переходу
    fn get_log_prob(&self, prev_word: &str, current_word: &str) -> f64 {
        // Для першого слова використовуємо уніграмну ймовірність
        if prev_word == "<START>" {
            return self.model.unigram_log_probs.get(current_word).copied().unwrap_or_else(|| SMOOTHING.log10());
        }
        // Для наступних слів - біграмну
        self.model.bigram_log_probs
            .get(prev_word)
            .and_then(|followers| followers.get(current_word))
            .copied()
            .unwrap_or_else(|| SMOOTHING.log10()) // Згладжування для невідомих пар
    }

    pub fn restore_text(&self, text: &str) -> String {
        let text = text.to_lowercase().replace([' ', '\n', '\r'], "");
        let n = text.len();
        
        let mut chart: Vec<Option<(f64, Vec<String>)>> = vec![None; n + 1];
        chart[0] = Some((0.0, Vec::new()));

        for i in 1..=n {
            let mut best_path_for_i: Option<(f64, Vec<String>)> = None;

            for j in 0..i {
                let len = i - j;
                if len > MAX_WORD_LEN { continue; }

                let segment = &text[j..i];
                if let Some((prev_prob, prev_path)) = &chart[j] {
                    if let Some(candidates) = self.model.dictionary_by_len.get(&len) {
                        
                        let best_candidate = candidates
                            .par_iter()
                            .filter(|&word| Self::is_match(segment, word))
                            .map(|word| {
                                let prev_word = prev_path.last().map_or("<START>", |s| s.as_str());
                                let log_p = self.get_log_prob(prev_word, word);
                                (log_p, word)
                            })
                            .max_by(|(p1, _), (p2, _)| p1.partial_cmp(p2).unwrap_or(std::cmp::Ordering::Equal));
                        
                        if let Some((log_p, word)) = best_candidate {
                            let new_prob = prev_prob + log_p;
                            if best_path_for_i.is_none() || new_prob > best_path_for_i.as_ref().unwrap().0 {
                                let mut new_path = prev_path.clone();
                                new_path.push(word.clone());
                                best_path_for_i = Some((new_prob, new_path));
                            }
                        }
                    }
                }
            }
            chart[i] = best_path_for_i;
        }

        chart[n].as_ref().map_or_else(
            || "Failed to restore text.".to_string(),
            |(_, path)| path.join(" "),
        )
    }
}

fn main() {
    let input_text = "Al*cew*sbegninnigtoegtver*triedofsitt*ngbyh*rsitsreonhtebnakandofh*vingnothi*gtodoonc*ortw*cesh*hdapee*edintoth*boo*h*rsiste*wasr*adnigbuti*hadnopictu*esorc*nve*sati*nsinitandwhatisth*useofab**kth*ughtAlic*withou*pic*u*esorco*versa*ions";

    println!("Starting text restoration with Viterbi algorithm...");
    let start_time = Instant::now();

    match LanguageModel::new("data/dictionary.txt", "data/corpus.txt") {
        Ok(model) => {
            let setup_duration = start_time.elapsed();
            println!("Decoder setup took: {:?}", setup_duration);

            let decoder = ViterbiDecoder::new(&model);
            
            let restore_start_time = Instant::now();
            let restored_text = decoder.restore_text(input_text);
            let restore_duration = restore_start_time.elapsed();
            
            println!("\n--- RESTORATION COMPLETE ---");
            println!("\nInput:\n{}", input_text);
            println!("\nOutput:\n{}", restored_text);
            println!("\nRestoration process took: {:?}", restore_duration);
            println!("Total time: {:?}", start_time.elapsed());
        }
        Err(e) => {
            eprintln!("\nError: Failed to initialize decoder: {}", e);
        }
    }
}