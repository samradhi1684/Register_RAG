"""HTML report -> PDF with WeasyPrint. Usage: python scripts/report_pdf.py <report.html> scripts/report_print.css <out.pdf>"""
import sys, weasyprint
src, css, out = sys.argv[1:4]
s = open(src, encoding="utf-8").read()
attr = {'band':'fill="#eef2f5"','grid':'stroke="#d9e0e6"','zero':'stroke="#72808d"','ci pos':'stroke="#1e7a46" stroke-width="2"','ci neg':'stroke="#b3363a" stroke-width="2"','ci nul':'stroke="#72808d" stroke-width="2"',
 'dot pos':'fill="#1e7a46"','dot neg':'fill="#b3363a"','dot nul':'fill="#ffffff" stroke="#72808d" stroke-width="2"','dot s7':'fill="#257de0"','dot s3':'fill="#f47230"','bar7':'fill="#257de0"','bar3':'fill="#f47230"','link':'stroke="#72808d" stroke-width="2"',
 'tick':'fill="#72808d" font-size="9"','tick dim':'fill="#72808d" font-size="8.5"','rowlab':'fill="#4a5866" font-size="9.5"','rowlab hl':'fill="#16202b" font-size="9.5" font-weight="700"','axlab':'fill="#72808d" font-size="9"','vlab':'fill="#16202b" font-size="10" font-weight="600"',
 'seg pos':'fill="#e3f3e9" stroke="#1e7a46"','seg nul':'fill="#eef2f5" stroke="#72808d"','seg neg':'fill="#fbe7e7" stroke="#b3363a"',
 'ln s7':'stroke="#257de0" stroke-width="2"','ln s3':'stroke="#f47230" stroke-width="2"','ln g':'stroke="#72808d" stroke-width="2"','dot g':'fill="#72808d"'}
for c, at in attr.items():
    s = s.replace(f'class="{c}"', f'class="{c}" {at}')
doc = '<!doctype html><html><head><meta charset="utf-8">' + s + "<style>" + open(css).read() + "</style></html>"
weasyprint.HTML(string=doc, base_url=".").write_pdf(out)
