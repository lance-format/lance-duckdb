//! Writing values as the Arrow types a Lance table stores.
//!
//! DuckDB exports a column with the Arrow type of its DuckDB type, which need
//! not be the type the table records: pyarrow-based writers store
//! `large_string`, dictionaries, `date64` or coarser time units, and DuckDB
//! labels TIMESTAMPTZ with the session time zone. Lance stores a column only
//! as its recorded type, so writes cast to it, but only between Arrow types
//! DuckDB reads as the same SQL type; anything else is still Lance's schema
//! error. A narrowing cast is kept only when it casts back to its input:
//! arrow-cast truncates when it narrows a time unit, and its default options
//! turn values it cannot convert into NULL.

use std::sync::Arc;

use arrow::compute::{cast_with_options, CastOptions};
use arrow::util::display::array_value_to_string;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use lance_core::datatypes::Field as LanceField;

/// Whether a write must cast `field` to `stored`: Lance records them as
/// different types that DuckDB reads as the same SQL type. Nested types pair
/// struct children by name and list, map and fixed-size list items in place,
/// as Lance's append check does. Fields carrying an Arrow extension type
/// (JSON, Blob v2, UUID, ...) are left to the existing extension handling.
pub(super) fn needs_stored_cast(field: &Field, stored: &Field) -> bool {
    representation_change(field, stored) == Some(true)
}

/// `Some(true)` when only the Arrow representation of the same SQL type
/// differs, `Some(false)` when Lance sees the same type, `None` otherwise.
fn representation_change(field: &Field, stored: &Field) -> Option<bool> {
    if field.extension_type_name().is_some() || stored.extension_type_name().is_some() {
        return field
            .data_type()
            .equals_datatype(stored.data_type())
            .then_some(false);
    }
    match (field.data_type(), stored.data_type()) {
        (
            DataType::List(item) | DataType::LargeList(item),
            DataType::List(stored_item) | DataType::LargeList(stored_item),
        ) => {
            let same_kind = std::mem::discriminant(field.data_type())
                == std::mem::discriminant(stored.data_type());
            Some(representation_change(item, stored_item)? || !same_kind)
        }
        (DataType::FixedSizeList(item, size), DataType::FixedSizeList(stored_item, stored_size))
            if size == stored_size =>
        {
            representation_change(item, stored_item)
        }
        (DataType::Map(entries, _), DataType::Map(stored_entries, _)) => {
            representation_change(entries, stored_entries)
        }
        (DataType::Struct(children), DataType::Struct(stored_children)) => {
            if children.len() != stored_children.len() {
                return None;
            }
            let mut changed = false;
            for child in children {
                let stored_child = stored_children.iter().find(|s| s.name() == child.name())?;
                changed |= representation_change(child, stored_child)?;
            }
            Some(changed)
        }
        (incoming, stored_type) => {
            if same_lance_type(field, stored) {
                Some(false)
            } else {
                let family = sql_family(incoming)?;
                (sql_family(stored_type)? == family).then_some(true)
            }
        }
    }
}

fn same_lance_type(field: &Field, stored: &Field) -> bool {
    matches!(
        (LanceField::try_from(field), LanceField::try_from(stored)),
        (Ok(incoming), Ok(stored)) if incoming.logical_type == stored.logical_type
    )
}

#[derive(PartialEq)]
enum SqlFamily {
    Varchar,
    Blob,
    Date,
    Time,
    Timestamp,
    TimestampTz,
    Interval,
}

/// The DuckDB SQL type an Arrow leaf type is read as, for the types that
/// have more than one Arrow representation.
fn sql_family(data_type: &DataType) -> Option<SqlFamily> {
    Some(match data_type {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => SqlFamily::Varchar,
        DataType::Dictionary(_, values) if sql_family(values) == Some(SqlFamily::Varchar) => {
            SqlFamily::Varchar
        }
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => SqlFamily::Blob,
        DataType::Date32 | DataType::Date64 => SqlFamily::Date,
        DataType::Time32(_) | DataType::Time64(_) => SqlFamily::Time,
        DataType::Timestamp(_, None) => SqlFamily::Timestamp,
        DataType::Timestamp(_, Some(_)) => SqlFamily::TimestampTz,
        DataType::Duration(_) | DataType::Interval(_) => SqlFamily::Interval,
        _ => return None,
    })
}

/// Whether casting `from` to `to` can change a value without failing:
/// arrow-cast divides when it narrows a time unit and drops the sub-unit part
/// of an interval it turns into a duration.
fn may_truncate(from: &DataType, to: &DataType) -> bool {
    fn rank(unit: &TimeUnit) -> u8 {
        match unit {
            TimeUnit::Second => 0,
            TimeUnit::Millisecond => 1,
            TimeUnit::Microsecond => 2,
            TimeUnit::Nanosecond => 3,
        }
    }
    match (from, to) {
        (DataType::Timestamp(from, _), DataType::Timestamp(to, _))
        | (DataType::Duration(from), DataType::Duration(to))
        | (
            DataType::Time32(from) | DataType::Time64(from),
            DataType::Time32(to) | DataType::Time64(to),
        ) => rank(to) < rank(from),
        (DataType::Interval(_), DataType::Duration(_)) | (DataType::Date64, DataType::Date32) => {
            true
        }
        (
            DataType::List(item) | DataType::LargeList(item) | DataType::FixedSizeList(item, _),
            DataType::List(to_item) | DataType::LargeList(to_item) | DataType::FixedSizeList(to_item, _),
        )
        | (DataType::Map(item, _), DataType::Map(to_item, _)) => {
            may_truncate(item.data_type(), to_item.data_type())
        }
        (DataType::Struct(children), DataType::Struct(to_children)) => {
            children.iter().any(|child| {
                to_children
                    .iter()
                    .find(|to_child| to_child.name() == child.name())
                    .is_some_and(|to_child| may_truncate(child.data_type(), to_child.data_type()))
            })
        }
        _ => false,
    }
}

/// `field` restated with the type and metadata the destination records.
pub(super) fn stored_field(field: &Field, stored: &Field) -> Field {
    Field::new(field.name(), stored.data_type().clone(), field.is_nullable())
        .with_metadata(stored.metadata().clone())
}

/// Casts every column of `batch` that `needs_stored_cast` selects to its
/// field in `stored`.
pub(super) fn cast_batch_to_stored(
    batch: &RecordBatch,
    stored: &Schema,
) -> Result<RecordBatch, String> {
    let schema = batch.schema();
    let mut fields = schema.fields().to_vec();
    let mut columns = batch.columns().to_vec();
    let mut changed = false;
    for (idx, field) in schema.fields().iter().enumerate() {
        let Ok(stored) = stored.field_with_name(field.name()) else {
            continue;
        };
        if !needs_stored_cast(field, stored) {
            continue;
        }
        columns[idx] = cast_to_stored(field.name(), &columns[idx], stored.data_type())?;
        fields[idx] = Arc::new(stored_field(field, stored));
        changed = true;
    }
    if !changed {
        return Ok(batch.clone());
    }
    let schema = Schema::new_with_metadata(fields, schema.metadata().clone());
    RecordBatch::try_new(Arc::new(schema), columns).map_err(|err| err.to_string())
}

/// Casts `array`, the values written to `column`, to `stored`. Fails
/// instead of storing a different value: the cast must succeed without
/// producing NULLs, and a narrowing cast must cast back to `array`.
pub(super) fn cast_to_stored(
    column: &str,
    array: &ArrayRef,
    stored: &DataType,
) -> Result<ArrayRef, String> {
    if array.data_type() == stored {
        return Ok(array.clone());
    }
    let options = CastOptions {
        safe: false,
        ..Default::default()
    };
    let error = |detail: String| {
        format!(
            "cannot write {} to column '{column}', which stores {stored}: {detail}",
            array.data_type()
        )
    };
    let cast = cast_with_options(array, stored, &options).map_err(|err| error(err.to_string()))?;
    if !may_truncate(array.data_type(), stored) {
        return Ok(cast);
    }
    let restored = cast_with_options(&cast, array.data_type(), &options)
        .map_err(|err| error(err.to_string()))?;
    if restored.as_ref() != array.as_ref() {
        let row = (0..array.len())
            .find(|&row| array.slice(row, 1).as_ref() != restored.slice(row, 1).as_ref())
            .unwrap_or_default();
        let value = array_value_to_string(array, row).unwrap_or_default();
        return Err(error(format!(
            "{value} would be stored as a different value"
        )));
    }
    Ok(cast)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::types::{
        Int32Type, IntervalMonthDayNano, IntervalMonthDayNanoType, TimestampMicrosecondType,
    };
    use arrow_array::{
        BinaryArray, Date32Array, DictionaryArray, ListArray, PrimitiveArray, StringArray,
        Time64MicrosecondArray, TimestampMicrosecondArray,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::TimeUnit;

    fn field(data_type: DataType) -> Field {
        Field::new("c", data_type, true)
    }

    fn timestamps(values: Vec<i64>, tz: &str) -> ArrayRef {
        Arc::new(TimestampMicrosecondArray::from(values).with_timezone(tz))
    }

    fn intervals(values: Vec<IntervalMonthDayNano>) -> ArrayRef {
        Arc::new(PrimitiveArray::<IntervalMonthDayNanoType>::from(values))
    }

    #[test]
    fn compares_like_lance() {
        let utf8 = field(DataType::Utf8);
        assert!(needs_stored_cast(&utf8, &field(DataType::LargeUtf8)));
        assert!(!needs_stored_cast(&utf8, &utf8));
        let tz = |tz: &str| field(DataType::Timestamp(TimeUnit::Microsecond, Some(tz.into())));
        assert!(needs_stored_cast(&tz("Asia/Shanghai"), &tz("UTC")));
        // List child names are restated by the append alignment, not cast.
        let list = |name: &str| {
            field(DataType::List(Arc::new(Field::new(
                name,
                DataType::Int64,
                true,
            ))))
        };
        assert!(!needs_stored_cast(&list("l"), &list("item")));
        let large_list = field(DataType::LargeList(Arc::new(Field::new(
            "item",
            DataType::LargeUtf8,
            true,
        ))));
        let utf8_list = field(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Utf8,
            true,
        ))));
        assert!(needs_stored_cast(&utf8_list, &large_list));
        let strukt = |a: DataType| {
            field(DataType::Struct(
                vec![
                    Field::new("a", a, true),
                    Field::new("b", DataType::Int64, true),
                ]
                .into(),
            ))
        };
        assert!(needs_stored_cast(
            &strukt(DataType::Utf8),
            &strukt(DataType::LargeUtf8)
        ));
        assert!(!needs_stored_cast(
            &strukt(DataType::Utf8),
            &strukt(DataType::Utf8)
        ));
        // Text is never recast into JSONB bytes, at any depth.
        let json = lance_arrow::json::json_field("c", true);
        assert!(!needs_stored_cast(&utf8, &json));
        let arrow_json = Field::new("j", DataType::Utf8, true).with_metadata(
            [(
                arrow_schema::extension::EXTENSION_TYPE_NAME_KEY.to_string(),
                lance_arrow::json::ARROW_JSON_EXT_NAME.to_string(),
            )]
            .into(),
        );
        let with_json = |j: Field, ts: &str| {
            field(DataType::Struct(
                vec![
                    j,
                    Field::new(
                        "ts",
                        DataType::Timestamp(TimeUnit::Microsecond, Some(ts.into())),
                        true,
                    ),
                ]
                .into(),
            ))
        };
        assert!(!needs_stored_cast(
            &with_json(arrow_json, "Asia/Shanghai"),
            &with_json(lance_arrow::json::json_field("j", true), "UTC")
        ));
        // Values DuckDB reads as another SQL type are left to Lance's check.
        let ts = |tz: Option<&str>| field(DataType::Timestamp(TimeUnit::Microsecond, tz.map(Into::into)));
        assert!(!needs_stored_cast(&field(DataType::Int64), &ts(None)));
        assert!(!needs_stored_cast(&ts(None), &ts(Some("UTC"))));
        assert!(!needs_stored_cast(&field(DataType::Int32), &field(DataType::Int64)));
        assert!(!needs_stored_cast(&field(DataType::Int32), &field(DataType::Date32)));
    }

    #[test]
    fn widens_and_relabels_values() {
        let strings: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None]));
        let cast = cast_to_stored("c", &strings, &DataType::LargeUtf8).unwrap();
        assert_eq!(cast.data_type(), &DataType::LargeUtf8);
        assert_eq!(cast.null_count(), 1);

        let dictionary_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let cast = cast_to_stored("c", &strings, &dictionary_type).unwrap();
        assert_eq!(cast.data_type(), &dictionary_type);

        let shanghai = timestamps(vec![1_704_067_200_000_000], "Asia/Shanghai");
        let utc = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        let cast = cast_to_stored("c", &shanghai, &utc).unwrap();
        let cast = cast
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(cast.value(0), 1_704_067_200_000_000);

        let dates: ArrayRef = Arc::new(Date32Array::from(vec![19_723]));
        let cast = cast_to_stored("c", &dates, &DataType::Date64).unwrap();
        assert_eq!(cast.data_type(), &DataType::Date64);
    }

    #[test]
    fn narrows_exact_values_only() {
        let utc_ms = DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()));
        assert!(cast_to_stored("c", &timestamps(vec![1_123_000], "UTC"), &utc_ms).is_ok());
        let err = cast_to_stored("c", &timestamps(vec![1_123_000, 1_123_456], "UTC"), &utc_ms)
            .unwrap_err();
        assert!(
            err.contains("'c'") && err.contains("different value"),
            "{err}"
        );

        let time32_s = DataType::Time32(TimeUnit::Second);
        let times =
            |micros: i64| -> ArrayRef { Arc::new(Time64MicrosecondArray::from(vec![micros])) };
        assert!(cast_to_stored("c", &times(1_000_000), &time32_s).is_ok());
        assert!(cast_to_stored("c", &times(1_900_000), &time32_s).is_err());

        let seconds = DataType::Duration(TimeUnit::Second);
        let interval =
            |days: i32, nanos: i64| intervals(vec![IntervalMonthDayNano::new(0, days, nanos)]);
        assert!(cast_to_stored("c", &interval(0, 2_000_000_000), &seconds).is_ok());
        assert!(cast_to_stored("c", &interval(0, 1_500_000_000), &seconds).is_err());
        assert!(cast_to_stored("c", &interval(1, 0), &seconds).is_err());
    }

    #[test]
    fn rejects_values_the_stored_type_cannot_hold() {
        let bytes: ArrayRef = Arc::new(BinaryArray::from(vec![&b"\x01\x02\x03\x04"[..]]));
        assert!(cast_to_stored("c", &bytes, &DataType::FixedSizeBinary(4)).is_ok());
        let bytes: ArrayRef = Arc::new(BinaryArray::from(vec![
            &b"\x01\x02\x03\x04\x05\x06\x07"[..],
        ]));
        assert!(cast_to_stored("c", &bytes, &DataType::FixedSizeBinary(4)).is_err());

        let keys = PrimitiveArray::<Int32Type>::from(vec![0, 1]);
        let values: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let dictionary: ArrayRef = Arc::new(DictionaryArray::try_new(keys, values).unwrap());
        assert!(cast_to_stored("c", &dictionary, &DataType::Utf8).is_ok());
    }

    #[test]
    fn checks_nested_values() {
        let item = |unit: TimeUnit| {
            Arc::new(Field::new(
                "item",
                DataType::Timestamp(unit, Some("UTC".into())),
                true,
            ))
        };
        let values =
            PrimitiveArray::<TimestampMicrosecondType>::from(vec![1_123_456]).with_timezone("UTC");
        let list: ArrayRef = Arc::new(ListArray::new(
            item(TimeUnit::Microsecond),
            OffsetBuffer::from_lengths([1]),
            Arc::new(values),
            None,
        ));
        let stored = DataType::List(item(TimeUnit::Millisecond));
        assert!(cast_to_stored("c", &list, &stored).is_err());
    }
}