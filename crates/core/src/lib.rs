// crates/core/src/lib.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

pub mod audio;
pub mod events;
pub mod result;

pub use audio::*;
pub use events::*;
pub use result::*;