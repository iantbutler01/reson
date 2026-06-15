#[cfg(all(feature = "postgres", feature = "tokio"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

#[cfg(not(all(feature = "postgres", feature = "tokio")))]
fn main() {
    eprintln!(
        "Run with: cargo run --example postgres_local_object_store --features postgres,tokio"
    );
}

#[cfg(all(feature = "postgres", feature = "tokio"))]
async fn run() -> Result<(), Box<dyn std::error::Error>> {
    use std::{env, path::PathBuf, sync::Arc};

    use bytes::Bytes;
    use chevalier_vfs::{
        OptimizedVfsStorage,
        index::VfsIndexScope,
        object_storage::{ObjectBackedVfsStorage, ObjectBackedVfsStorageConfig},
        object_store::LocalObjectStoreClient,
        postgres_index::PostgresVfsManifestIndex,
    };
    use sqlx::PgPool;

    let database_url = env::var("CHEVALIER_VFS_DATABASE_URL")?;
    let object_root = env::var("CHEVALIER_VFS_OBJECT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".run/chevalier-vfs-objects"));
    let scope = env::var("CHEVALIER_VFS_SCOPE").unwrap_or_else(|_| "dev-scope".to_string());

    let pool = PgPool::connect(&database_url).await?;
    let index = Arc::new(PostgresVfsManifestIndex::new(pool));
    let store = Arc::new(LocalObjectStoreClient::new(object_root)?);
    let vfs = ObjectBackedVfsStorage::new(
        ObjectBackedVfsStorageConfig::new(VfsIndexScope::new(scope)),
        store,
        index,
    );

    vfs.mkdir("notes").await?;
    vfs.write(
        "notes/hello.txt",
        Bytes::from_static(b"hello from chevalier-vfs"),
        None,
    )
    .await?;
    let bytes = vfs.read("notes/hello.txt").await?;
    println!("{}", String::from_utf8_lossy(&bytes));
    Ok(())
}
