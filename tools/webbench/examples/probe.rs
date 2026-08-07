fn main() {
    for src in ["<p>one<p>two", "<ul><li>a<li>b</ul>", "<table><tr><td>x</table>"] {
        let dom = tl::parse(src, tl::ParserOptions::default()).unwrap();
        let p = dom.parser();
        println!("--- {src}");
        fn walk(h: &tl::NodeHandle, p: &tl::Parser, d: usize) {
            let Some(n) = h.get(p) else { return };
            if let Some(t) = n.as_tag() {
                println!("{}<{}>", "  ".repeat(d), t.name().as_utf8_str());
            }
            if let Some(k) = n.children() {
                for c in k.top().iter() { walk(c, p, d + 1); }
            }
        }
        for h in dom.children() { walk(h, p, 0); }
    }
    // Does the Google script parse under boa but not us?
    if let Ok(s) = std::fs::read_to_string(std::env::args().nth(1).unwrap_or_default()) {
        let mut i = boa_interner::Interner::default();
        let boa_ok = boa_parser::Parser::new(boa_engine::Source::from_bytes(s.as_bytes()))
            .parse_script(&boa_ast::scope::Scope::new_global(), &mut i).is_ok();
        let just_ok = just_engine::parser::JsParser::parse_to_ast_from_str(&s).is_ok();
        println!("--- google inline script ({} B): just={just_ok} boa={boa_ok}", s.len());
    }
}
