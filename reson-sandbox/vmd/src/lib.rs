#![allow(clippy::clone_on_copy)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_find)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::unnecessary_cast)]

pub mod app;
pub mod assets;
pub mod bootstrap;
pub mod config;
pub mod image;
pub mod state;
pub mod virt;

pub mod proto {
    pub mod v1 {
        include!(concat!(env!("OUT_DIR"), "/vmd.v1.rs"));
    }
}
