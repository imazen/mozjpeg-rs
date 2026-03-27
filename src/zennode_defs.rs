//! zennode pipeline node definitions for mozjpeg-rs.
//!
//! Provides [`EncodeMozjpeg`] node with full parameter schema,
//! RIAPI querystring parsing, and conversion to
//! [`MozjpegEncoderConfig`](crate::codec::MozjpegEncoderConfig).
//!
//! Feature-gated behind `feature = "zennode"`.

use zennode::*;

use crate::codec::MozjpegEncoderConfig;
use crate::types::Subsampling;

// ============================================================================
// EncodeMozjpeg node
// ============================================================================

/// mozjpeg encoder configuration as a self-documenting pipeline node.
///
/// Maps to [`MozjpegEncoderConfig`] via
/// [`to_encoder_config()`](EncodeMozjpeg::to_encoder_config).
///
/// **RIAPI**: `?mozjpeg.quality=85&mozjpeg.effort=2&mozjpeg.subsampling=420`
#[derive(Node, Clone, Debug, Default)]
#[node(id = "mozjpeg-rs.encode", group = Encode, role = Encode)]
#[node(tags("jpeg", "jpg", "encode", "lossy", "mozjpeg"))]
pub struct EncodeMozjpeg {
    /// Quality level (1-100, mozjpeg native scale). None = use default (85).
    #[param(range(1.0..=100.0), default = 85.0, step = 1.0)]
    #[param(section = "Quality", label = "Quality")]
    #[kv("mozjpeg.quality", "mozjpeg.q")]
    pub quality: Option<f32>,

    /// Encoding effort (0 = fastest baseline, 1 = baseline balanced,
    /// 2 = progressive balanced, 3 = progressive smallest).
    /// None = use default (2).
    #[param(range(0..=3), default = 2)]
    #[param(section = "Quality", label = "Effort")]
    #[kv("mozjpeg.effort")]
    pub effort: Option<i32>,

    /// Chroma subsampling: "444", "422", "420", "440", "gray".
    /// None = use default (420).
    #[param(default = "420")]
    #[param(section = "Color", label = "Chroma Subsampling")]
    #[kv("mozjpeg.subsampling", "mozjpeg.ss")]
    pub subsampling: Option<String>,
}

impl EncodeMozjpeg {
    /// Apply this node's explicitly-set params on top of an existing
    /// [`MozjpegEncoderConfig`].
    ///
    /// `None` fields are skipped, so the base config's values are preserved.
    pub fn apply(&self, mut config: MozjpegEncoderConfig) -> MozjpegEncoderConfig {
        use zencodec::encode::EncoderConfig as _;

        if let Some(quality) = self.quality {
            config = config.with_generic_quality(quality);
        }
        if let Some(effort) = self.effort {
            config = config.with_generic_effort(effort);
        }
        if let Some(ref ss) = self.subsampling
            && let Some(subsampling) = Self::parse_subsampling(ss)
        {
            config = config.with_subsampling(subsampling);
        }
        config
    }

    /// Convert this node into a new [`MozjpegEncoderConfig`] using defaults
    /// for unset fields.
    pub fn to_encoder_config(&self) -> MozjpegEncoderConfig {
        self.apply(MozjpegEncoderConfig::new())
    }

    /// Parse a subsampling string into a [`Subsampling`] value.
    fn parse_subsampling(s: &str) -> Option<Subsampling> {
        Some(match s.to_ascii_lowercase().as_str() {
            "444" | "none" | "4:4:4" => Subsampling::S444,
            "422" | "4:2:2" | "half_horizontal" => Subsampling::S422,
            "420" | "4:2:0" | "quarter" => Subsampling::S420,
            "440" | "4:4:0" | "half_vertical" => Subsampling::S440,
            "gray" | "grayscale" => Subsampling::Gray,
            _ => return None,
        })
    }
}

/// Register all mozjpeg-rs zennode definitions with a registry.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(&ENCODE_MOZJPEG_NODE);
}

/// All mozjpeg-rs zennode definitions.
pub static ALL: &[&dyn NodeDef] = &[&ENCODE_MOZJPEG_NODE];

#[cfg(test)]
mod tests {
    use super::*;
    use zencodec::encode::EncoderConfig as _;

    #[test]
    fn schema_exists() {
        let schema = ENCODE_MOZJPEG_NODE.schema();
        assert_eq!(schema.id, "mozjpeg-rs.encode");
        assert_eq!(schema.group, NodeGroup::Encode);
        assert!(!schema.params.is_empty());
    }

    #[test]
    fn default_values() {
        let node = EncodeMozjpeg::default();
        assert_eq!(node.quality, None);
        assert_eq!(node.effort, None);
        assert_eq!(node.subsampling, None);
    }

    #[test]
    fn to_encoder_config_defaults() {
        let node = EncodeMozjpeg::default();
        let config = node.to_encoder_config();
        assert_eq!(config.generic_quality(), Some(85.0));
        assert_eq!(config.generic_effort(), Some(2));
    }

    #[test]
    fn to_encoder_config_custom() {
        let node = EncodeMozjpeg {
            quality: Some(50.0),
            effort: Some(0),
            subsampling: Some("444".into()),
        };
        let config = node.to_encoder_config();
        assert_eq!(config.generic_quality(), Some(50.0));
        assert_eq!(config.generic_effort(), Some(0));
    }

    #[test]
    fn riapi_parsing() {
        let mut kv = KvPairs::from_querystring("mozjpeg.quality=75&mozjpeg.effort=3");
        let instance = ENCODE_MOZJPEG_NODE.from_kv(&mut kv).unwrap().unwrap();
        let node = instance.as_any().downcast_ref::<EncodeMozjpeg>().unwrap();
        assert_eq!(node.quality, Some(75.0));
        assert_eq!(node.effort, Some(3));
    }

    #[test]
    fn apply_preserves_unset() {
        let base = MozjpegEncoderConfig::new()
            .with_generic_quality(90.0)
            .with_generic_effort(1);

        // Only set quality — effort should be preserved from base
        let node = EncodeMozjpeg {
            quality: Some(50.0),
            effort: None,
            subsampling: None,
        };
        let config = node.apply(base);
        assert_eq!(config.generic_quality(), Some(50.0));
        assert_eq!(config.generic_effort(), Some(1)); // preserved
    }

    #[test]
    fn registry_roundtrip() {
        let mut registry = NodeRegistry::default();
        register(&mut registry);
        assert!(registry.get("mozjpeg-rs.encode").is_some());
    }

    #[test]
    fn subsampling_parsing() {
        assert_eq!(
            EncodeMozjpeg::parse_subsampling("420"),
            Some(Subsampling::S420)
        );
        assert_eq!(
            EncodeMozjpeg::parse_subsampling("4:4:4"),
            Some(Subsampling::S444)
        );
        assert_eq!(
            EncodeMozjpeg::parse_subsampling("gray"),
            Some(Subsampling::Gray)
        );
        assert_eq!(EncodeMozjpeg::parse_subsampling("invalid"), None);
    }
}
