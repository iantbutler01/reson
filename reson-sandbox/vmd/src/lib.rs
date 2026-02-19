#![allow(clippy::clone_on_copy)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_find)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::unnecessary_cast)]

pub mod app;
pub mod assets;
pub mod bootstrap;
pub mod config;
pub mod control_bus;
pub mod image;
pub mod reconcile;
pub mod registry;
pub mod state;
pub mod virt;

pub mod proto {
    pub mod v1 {
        include!(concat!(env!("OUT_DIR"), "/vmd.v1.rs"));
    }
}
