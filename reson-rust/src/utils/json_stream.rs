//! Streaming JSON parsing utilities.
//!
//! Handles JSON values delivered in arbitrary chunk boundaries (including
//! JSON arrays with separators).

use crate::error::{Error, Result};
use serde::Deserialize;

pub fn parse_json_value_strict_str(
    input: &str,
) -> std::result::Result<serde_json::Value, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_str(input);
    let value = serde_json::Value::deserialize(&mut de)?;
    de.end()?;
    Ok(value)
}

pub fn parse_json_value_strict_bytes(
    input: &[u8],
) -> std::result::Result<serde_json::Value, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_slice(input);
    let value = serde_json::Value::deserialize(&mut de)?;
    de.end()?;
    Ok(value)
}

#[derive(Default)]
pub struct JsonStreamAccumulator {
    buffer: Vec<u8>,
}

impl JsonStreamAccumulator {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<Vec<serde_json::Value>> {
        self.buffer.extend_from_slice(bytes);

        let mut values = Vec::new();

        'parse: loop {
            let mut offset = 0;
            while offset < self.buffer.len() {
                match self.buffer[offset] {
                    b'[' | b']' | b',' | b'\n' | b'\r' | b' ' | b'\t' => offset += 1,
                    _ => break,
                }
            }
            if offset > 0 {
                self.buffer.drain(0..offset);
            }
            if self.buffer.is_empty() {
                break;
            }

            let deserializer = serde_json::Deserializer::from_slice(&self.buffer);
            let mut iter = deserializer.into_iter::<serde_json::Value>();
            let mut parsed_any = false;

            while let Some(value_result) = iter.next() {
                match value_result {
                    Ok(value) => {
                        parsed_any = true;
                        values.push(value);
                    }
                    Err(err) if err.is_eof() => {
                        break;
                    }
                    Err(err) => {
                        self.buffer.clear();
                        return Err(Error::Inference(format!(
                            "JSON stream parse error: {}",
                            err
                        )));
                    }
                }
            }

            let consumed = iter.byte_offset();
            if consumed > 0 {
                self.buffer.drain(0..consumed);
            }
            if !parsed_any {
                break;
            }
        }

        Ok(values)
    }
}
