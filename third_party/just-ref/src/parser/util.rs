// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::parser::ast::HasMeta;

pub const TAB_WIDTH: usize = 2;

pub struct StructBuilder<'a> {
    name: &'a str,
    fields: Vec<&'a str>,
    field_values: Vec<String>,
}

impl<'a> StructBuilder<'a> {
    fn new(name: &'a str) -> Self {
        StructBuilder {
            name,
            fields: Vec::new(),
            field_values: Vec::new(),
        }
    }

    pub fn add_fields(&'a mut self, field_name: &'a str, value: String) -> &'a mut StructBuilder<'a> {
        self.fields.push(field_name);
        self.field_values.push(value);
        self
    }

    pub fn to_string(&self) -> String {
        let mut s = format!("{} {{\n", self.name);
        let mut idx = 0;
        for f in &self.fields {
            s.push_str(
                format!(
                    "{}{}: {},\n",
                    spaces(TAB_WIDTH),
                    f,
                    indent_value(self.field_values.get(idx).unwrap())
                )
                .as_str(),
            );
            idx += 1;
        }
        s.push_str("}");
        s
    }
}

fn indent_value(v: &String) -> String {
    v.replace("\n", format!("\n{}", spaces(TAB_WIDTH)).as_str())
        .trim_end()
        .to_string()
}

pub fn spaces(time: usize) -> String {
    " ".repeat(time)
}

pub fn format_struct(struct_name: &str) -> StructBuilder<'_> {
    StructBuilder::new(struct_name)
}

pub fn format_vec<V, T: Fn(&V) -> String>(vec: &Vec<V>, unwrap: T) -> String {
    let mut s = String::new();
    s.push_str("[\n");
    for i in vec {
        s.push_str(format!("{}{},\n", spaces(TAB_WIDTH), indent_value(&unwrap(i))).as_str());
    }
    s.push_str("]");
    s
}

pub fn format_option<T, U: Fn(&T) -> String>(value: &Option<T>, on_unwrap: U) -> String {
    match value {
        Some(a) => on_unwrap(a),
        None => "<None>".to_string(),
    }
}

pub fn format_has_meta_option<T: HasMeta>(value: &Option<T>, script: &str) -> String {
    match value {
        Some(a) => a.to_formatted_string(script),
        None => "<None>".to_string(),
    }
}
