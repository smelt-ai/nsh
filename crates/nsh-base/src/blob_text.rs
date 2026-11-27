//! Large texts from the app to Rust components as blobs, so their bytes
//! never go through the app's heap: a prop takes one by a `blob:` URL.

use nsh_core::element::{JsHost, JsIn, JsOwned, JsValue, PropError, SharedString, TsTypes};

/// A text a prop takes by a `blob:` URL (`URL.createObjectURL`): the
/// blob's bytes, as UTF-8, read when the prop is set, so a large text
/// goes to the view without going through the app's heap (`patchUrl`,
/// `valueUrl`). A URL no blob has (revoked, never made) is an empty text.
#[derive(Debug, Clone, Default)]
pub struct BlobText {
    /// The URL.
    pub url: SharedString,
    /// The blob's text.
    pub text: SharedString,
}

impl PartialEq for BlobText {
    /// A blob's bytes never change: the same URL is the same text.
    fn eq(&self, other: &Self) -> bool {
        self.url == other.url
    }
}

impl JsValue for BlobText {
    fn ts_type(_types: &mut TsTypes) -> String {
        "string".into()
    }

    fn from_js(value: JsIn<'_>) -> Result<Self, PropError> {
        let url = value
            .as_string()
            .ok_or_else(|| value.expected("a blob: URL"))?;
        let text = match value.host().blob(&url) {
            Some((bytes, _)) => String::from_utf8_lossy(&bytes).into_owned(),
            None => {
                log::warn!("no blob has the URL {url}: its text is empty");
                String::new()
            }
        };
        Ok(BlobText {
            url: url.into(),
            text: text.into(),
        })
    }

    fn to_js<'a>(&self, host: &'a dyn JsHost) -> Result<JsOwned<'a>, PropError> {
        self.url.to_js(host)
    }
}
