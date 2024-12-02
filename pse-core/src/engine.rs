// use pyo3::prelude::*;
// use pyo3::types::{PyAny, PyDict};
// use std::collections::{HashMap, HashSet};
// use tokenizers::{Tokenizer, Encoding};
// use trie_rs::TrieBuilder;
// use log::{info, warn};

// use crate::walker::Walker;
// use crate::state_machine::StateMachine;
// use crate::acceptor::Acceptor;


// // Define any additional structs or enums you need
// #[pyclass]
// pub struct StructuringEngine {
//     tokenizer: Tokenizer,
//     acceptor: Option<Acceptor>,
//     walkers: Vec<Walker>,
//     within_json_value: bool,
//     vocabulary: HashMap<String, u32>,
//     reverse_vocabulary: HashMap<u32, String>,
// }

// #[pymethods]
// impl StructuringEngine {
//     #[new]
//     #[pyo3(signature = (tokenizer, vocabulary=None))]
//     pub fn new(tokenizer: Tokenizer, vocabulary: Option<HashMap<String, u32>>) -> PyResult<Self> {
//         let mut engine = StructuringEngine {
//             tokenizer,
//             acceptor: None,
//             walkers: Vec::new(),
//             within_json_value: false,
//             vocabulary: HashMap::new(),
//             reverse_vocabulary: HashMap::new(),
//         };
//         engine.build_vocabulary(tokenizer, vocabulary)?;
//         Ok(engine)
//     }

//     pub fn advance_token(&mut self, token_id: u32) -> PyResult<Option<u32>> {
//         let token = match self.reverse_vocabulary.get(&token_id) {
//             Some(t) => t.clone(),
//             None => {
//                 warn!("Unknown token ID: {}", token_id);
//                 return Ok(None);
//             }
//         };

//         let mut seen: HashMap<String, HashSet<Walker>> = HashMap::new();
//         let mut longest_partial: (String, u32) = (String::new(), 0);

//         let (new_walkers, valid_tokens) = StateMachine::advance_all(&self.walkers, &token, &self.dawg)?;

//         for (valid_token, walker) in new_walkers {
//             seen.entry(valid_token.clone()).or_insert_with(HashSet::new).insert(walker);

//             if valid_token != token {
//                 if valid_token.len() > longest_partial.0.len() {
//                     if let Some(&valid_id) = self.vocabulary.get(&valid_token) {
//                         longest_partial = (valid_token.clone(), valid_id);
//                     }
//                 }
//             }
//         }

//         if let Some(walkers) = seen.get(&token) {
//             self.walkers = walkers.iter().cloned().collect();
//             Ok(Some(token_id))
//         } else if longest_partial.1 != 0 {
//             if let Some(walkers) = seen.get(&longest_partial.0) {
//                 self.walkers = walkers.iter().cloned().collect();
//                 Ok(Some(longest_partial.1))
//             } else {
//                 Ok(None)
//             }
//         } else {
//             Ok(None)
//         }
//     }

//     pub fn get_valid_tokens(&self) -> PyResult<(HashSet<String>, Trie)> {
//         let mut all_valid_prefixes = HashSet::new();
//         let mut trie = Trie::new();

//         for walker in &self.walkers {
//             if walker.accepts_any_token()? {
//                 return Ok((HashSet::new(), trie));
//             }

//             let valid_prefixes = walker.find_valid_prefixes(&self.dawg)?;
//             all_valid_prefixes.extend(valid_prefixes);
//         }

//         for s in &all_valid_prefixes {
//             trie.add(&s.chars().rev().collect::<String>());
//         }

//         Ok((all_valid_prefixes, trie))
//     }

//     pub fn consume_raw_input(&mut self, raw_input: &str) -> PyResult<()> {
//         // Process each token of the raw string input
//         let token_ids = self.tokenizer.encode(raw_input, false)?.get_ids().to_vec();
//         for token_id in token_ids {
//             let token = self.tokenizer.decode(&[token_id], false)?;
//             if token.is_empty() {
//                 continue;
//             }

//             let (new_walkers, _) = StateMachine::advance_all(&self.walkers, &token, &self.dawg)?;
//             let walkers: Vec<Walker> = new_walkers.into_iter().filter(|(valid_token, _)| valid_token == &token).map(|(_, walker)| walker).collect();

//             if !walkers.is_empty() {
//                 self.walkers = walkers;
//             }
//         }
//         Ok(())
//     }

//     #[classmethod]
//     #[pyo3(signature = (tokenizer, vocabulary=None))]
//     pub fn build_vocabulary(_cls: &PyType, tokenizer: Tokenizer, vocabulary: Option<HashMap<String, u32>>) -> PyResult<()> {
//         let vocab = match vocabulary {
//             Some(v) => v,
//             None => {
//                 let py_vocab = tokenizer.get_vocab(true);
//                 py_vocab
//             }
//         };

//         let mut builder = TrieBuilder::new();

//         let decoded_tokens = match vocabulary {
//             Some(_) => vocab.keys().cloned().collect::<Vec<String>>(),
//             None => {
//                 let token_ids: Vec<u32> = vocab.values().cloned().collect();

//                 token_ids
//                     .iter()
//                     .map(|&id| {
//                         tokenizer
//                             .id_to_token(id)
//                             .map(|s| s.to_string())
//                             .ok_or_else(|| {
//                                 PyErr::new::<pyo3::exceptions::PyException, _>(format!(
//                                     "Unknown token ID: {}",
//                                     id
//                                 ))
//                             })
//                     })
//                     .collect::<Result<Vec<String>, PyErr>>()?
//             }
//         };

//         for token in &decoded_tokens {
//             builder.push(token);
//         }

//         let trie = builder.build();

//         let mut vocabulary = HashMap::new();
//         let mut reverse_vocabulary = HashMap::new();

//         for (token, id) in &decoded_tokens.iter().zip(token_ids.iter()) {
//             vocabulary.insert(token.clone(), *id);
//             reverse_vocabulary.insert(*id, token.clone());
//         }

//         // Assign to class variables
//         // Self::dawg = dawg;
//         Self::vocabulary = vocabulary;
//         Self::reverse_vocabulary = reverse_vocabulary;

//         Ok(())
//     }


// }
