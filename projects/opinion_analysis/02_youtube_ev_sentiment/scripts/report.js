const rows = JSON.parse(document.getElementById('comment-data').textContent);
const colors = {positive:'#238B8E',neutral:'#9CA3AF',negative:'#D76A4A'};
const labels = {positive:'正面',neutral:'中性',negative:'负面'};
const inputs = ['search','type-filter','sentiment-filter','topic-filter'].map(id=>document.getElementById(id));
const names = [...new Set(rows.map(r=>r.topic_name))];
names.forEach(name=>{ const option=document.createElement('option'); option.value=name; option.textContent=name; inputs[3].appendChild(option); });
let limit=20;
function render(){
 const query=inputs[0].value.toLowerCase();
 const filtered=rows.filter(r=>(r.text_raw.toLowerCase().includes(query)||r.comment_id.toLowerCase().includes(query))&&(!inputs[1].value||r.type===inputs[1].value)&&(!inputs[2].value||r.sentiment_label===inputs[2].value)&&(!inputs[3].value||r.topic_name===inputs[3].value));
 document.getElementById('search-count').textContent=`共 ${filtered.length} 条匹配 · 当前显示 ${Math.min(limit,filtered.length)} 条`;
 const container=document.getElementById('records');container.replaceChildren();
 filtered.slice(0,limit).forEach(r=>{
  const article=document.createElement('article'); article.className='record';
  const badge=document.createElement('span');badge.className='badge';badge.style.background=colors[r.sentiment_label];badge.textContent=labels[r.sentiment_label];
  const meta=document.createElement('span');meta.className='meta';meta.textContent=`${r.comment_id} · ${r.type==='comment'?'主评论':'回复'} · ${r.topic_name} · 最大概率 ${Number(r.confidence).toFixed(2)}`;
  const text=document.createElement('p');text.textContent=r.text_raw;article.append(badge,meta,text);container.append(article);
 });
 document.getElementById('more').hidden=limit>=filtered.length;
}
inputs.forEach(el=>el.addEventListener('input',()=>{limit=20;render();}));
document.getElementById('more').addEventListener('click',()=>{limit+=20;render();});
document.getElementById('semantic-map').on('plotly_click', event=>{
 const row=rows.find(r=>r.comment_id===event.points[0].customdata); if(!row)return;
 document.getElementById('selected-comment').textContent=`${row.comment_id} · ${row.type==='comment'?'主评论':'回复'} · ${row.topic_name} · 整体${labels[row.sentiment_label]}\n\n${row.text_raw}`;
});
render();
