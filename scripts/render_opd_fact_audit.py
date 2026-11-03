#!/usr/bin/env python3
"""Render the frozen factual audit as a self-contained local review artifact."""
import base64
import html
import io
import json
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"outputs/opd_fact_audit_20261019"
escape=lambda x:html.escape(str(x))


def main():
    cases={r["source_index"]:r for r in json.loads((FOLDER/"cases.json").read_text())}
    report=json.loads((FOLDER/"report.json").read_text())
    scores=[json.loads(s) for s in (FOLDER/"scores.jsonl").read_text().splitlines()]
    sections=[]
    for score in scores:
        row=cases[score["source_index"]]; ann=score["annotation"]
        images=[]
        for path in row["frames"]:
            image=Image.open(path).convert("RGB"); image.thumbnail((480,300))
            buffer=io.BytesIO(); image.save(buffer,format="JPEG",quality=90)
            images.append('<img src="data:image/jpeg;base64,%s">'%base64.b64encode(buffer.getvalue()).decode())
        facts=[]
        for fact in score["facts"]:
            lines=[]
            for phrase,prob in fact["alternatives"].items():
                lines.append("<tr><td>%s</td><td>%.6g</td><td>%.6g</td><td>%.3f</td></tr>"%(
                    escape(repr(phrase)),prob["plain_probability"],prob["reference_probability"],prob["logp_change"]))
            facts.append("<p>位置 %s · 原 token %s · 前缀状态 %s · 目标/对照 log-odds 变化 %.3f</p>"
                         "<table><tr><th>候选</th><th>普通提示概率</th><th>参考条件概率</th><th>logP 变化</th></tr>%s</table>"%(
                             fact["position"],escape(repr(fact["observed_token"])),escape(fact["prefix_state"]),fact["margin_change"],"".join(lines)))
        sections.append("<article><h2>#%s · %s</h2><p>%s</p><p>参考：%s</p><p>原学生：%s</p>"
                        "<p>教师重新生成：%s · Judge=%s</p><div class=frames>%s</div><p><strong>%s</strong>：%s</p>%s</article>"%(
                            score["source_index"],escape(row["cohort"]),escape(row["question"]),escape(row["answers"]),
                            escape(row["student_answer"]),escape(score["teacher_answer"]),escape(score["teacher_judge"]),
                            "".join(images),escape(ann["status"]),escape(ann["evidence"]),"".join(facts)))
    document="""<!doctype html><meta charset="utf-8"><title>OPD 教师事实监督审核</title>
<style>body{font:16px/1.5 system-ui;max-width:1650px;margin:32px auto;padding:0 20px;background:#f6f7f8;color:#17232d}
article{background:white;border:1px solid #ddd;border-radius:10px;padding:18px;margin:24px 0}.frames{display:flex;overflow:auto}.frames img{width:20%;min-width:240px;object-fit:contain}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:7px}pre{white-space:pre-wrap}</style>
<h1>reference_only 教师的关键事实监督审核</h1><p>32 个定向训练案例，31 个事实对比。由 Codex 核对图像，不是独立人工金标，也不是总体错误率抽样。
不确定及参考存疑案例单独保留；概率是原学生前缀下的候选词/短语概率，多 token 短语包含长度影响。
历史 Judge 阳性与视觉事实支持是两套不同标记。本页不修改训练标签和评测分数。</p>"""
    document+="<details><summary>汇总与文件指纹</summary><pre>"+escape(json.dumps(report,ensure_ascii=False,indent=2))+"</pre></details>"
    (FOLDER/"report.html").write_text(document+"".join(sections))
    print(FOLDER/"report.html")


if __name__=="__main__": main()
