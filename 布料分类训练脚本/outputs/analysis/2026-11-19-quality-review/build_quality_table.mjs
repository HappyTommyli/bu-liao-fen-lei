import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const dir=fileURLToPath(new URL('.',import.meta.url));
const t=JSON.parse(await fs.readFile(dir+'表格输入.json','utf8'));const d=JSON.parse(await fs.readFile(dir+'质量与去回一致性结果.json','utf8'));
const wb=Workbook.create();const summary=wb.worksheets.add('质量汇总');const selected=wb.worksheets.add('筛选清单');const all=wb.worksheets.add('全部采集');
const header={fill:'#244B69',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:44};
function letter(n){let s='';for(;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;}
function base(s,rows,cols){s.showGridLines=false;s.getRangeByIndexes(0,0,rows,cols).format.font={name:'Arial',size:11,color:'#243447'};s.getRangeByIndexes(0,0,rows,cols).format.verticalAlignment='center';}
base(summary,40,10);summary.tabColor='#244B69';summary.getRange('A2').values=[['布料采集质量与去回一致性 · 2026-11-19']];summary.getRange('A2').format.font={size:16,bold:true};
summary.getRange('A3').values=[[`本次固定扫描${d.statistics.total}个布料CSV、${d.statistics.classes}类；另10个轨迹测试不纳入。仅评估原始采集，不使用分类模型结果。`]];
summary.getRange('A4').values=[['541条通过，12条可用建议复核，38条按现行规则不合格，135条坐标/接触范围不兼容待核对。原始文件全部保留。']];
summary.getRange('A5').values=[['135条坐标待核对仍不能按当前旧协议训练；这里单列是避免把坐标不兼容误当成磁数据损坏。']];
summary.getRange('A7:J25').values=[t.summaryheaders,...t.summaryrows];summary.tables.add('A7:J25',true,'CollectionQualitySummary');summary.getRange('A7:J7').format=header;summary.getRange('A8:J25').format.rowHeight=31;
const widths=[25,14,13,22,15,15,26,29,56,50];widths.forEach((w,i)=>summary.getRangeByIndexes(6,i,20,1).format.columnWidth=w);
summary.getRange('B8:F26').setNumberFormat('0');summary.getRange('G8:H25').setNumberFormat('0.00%');summary.getRange('A26').values=[['合计']];
for(const col of ['B','C','D','E','F'])summary.getRange(col+'26').formulas=[[`=SUM(${col}8:${col}25)`]];summary.getRange('A26:F26').format.font={bold:true};
summary.getRange('E8:E25').conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0,format:{fill:'#FCEAE7',font:{color:'#A5362D'}}});
summary.getRange('F8:F25').conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0,format:{fill:'#FFF1D9',font:{color:'#805A13'}}});
const notes=[
 ['A28','不合格38条：历史确认排除7条、四项去回不一致全部满足3条、接触窗内采样缺口超过20ms共28条。'],
 ['A29','暂缓135条：122条软件X整体偏移约−24.35mm、13条约+26mm，往返行程仍约155mm；先核对机械基准与坐标关系。'],
 ['A30','若假设仅坐标零点平移，135条均未触发幅值四条件；126条未见旧接触窗内超20ms缺口，另9条仍有缺口。此假设不解除准入限制。'],
 ['A31','完整窗：去程45～142mm、回程16～114mm，边界容差2mm；采样333Hz、最大缺口20ms、差分/基线容差0.01μT。'],
 ['A32','去回四条件：两种范围差异均≥10%、偏差同向、幅值差≥max(5μT,3倍基线噪声)、至少80%共同位置箱同向。'],
 ['A33','共同X按有效采样中位数；2mm位置箱、每箱每方向至少3点、至少10个共同箱。相关系数仅展示，不单凭相关低排除。'],
 ['A34','可用记录去回差异：中位数1.72%，90分位4.68%，最大9.86%。坐标不兼容记录的原窗口指标仅供描述。'],
 ['A35','筛选清单含不合格、暂缓与可用建议复核；按“处理建议”过滤即可。CSV/metadata双指纹识别原记录，同名复采重新审查。'],
 ['A37','数据源：'+d.scope.source],
 ['A38','完整规则版本：'+d.rule.version+'；完整审查批次与原始文件指纹保存在本次报告目录。']
];
for(const [cell,value]of notes)summary.getRange(cell).values=[[value]];summary.getRange('A28:A38').format.font={size:11,color:'#596C7C'};summary.getRange('A28:J38').format.rowHeight=25;summary.freezePanes.freezeRows(7);summary.freezePanes.freezeColumns(1);
function details(sheet,rows,name){const n=rows.length+1;const c=t.headers.length;const end=letter(c);base(sheet,n,c);sheet.getRangeByIndexes(0,0,n,c).values=[t.headers,...rows];sheet.tables.add(`A1:${end}${n}`,true,name);sheet.getRangeByIndexes(0,0,1,c).format=header;sheet.getRangeByIndexes(1,0,rows.length,c).format.rowHeight=27;
 const widths=[25,34,31,32,24,24,24,24,26,26,26,26,26,26,25,25,90,85,40,40,72,72,140,45,45,32,35,38,38];widths.forEach((w,i)=>sheet.getRangeByIndexes(0,i,n,1).format.columnWidth=w);
 for(const col of ['E','F','K','L','P','Z','AB','AC'])sheet.getRange(`${col}2:${col}${n}`).setNumberFormat('0.00');for(const col of ['I','J','M','O','AA'])sheet.getRange(`${col}2:${col}${n}`).setNumberFormat('0.00%');
 sheet.getRange(`C2:C${n}`).conditionalFormats.add('containsText',{text:'不合格',format:{fill:'#FCEAE7',font:{color:'#A5362D'}}});sheet.getRange(`C2:C${n}`).conditionalFormats.add('containsText',{text:'暂缓',format:{fill:'#FFF1D9',font:{color:'#805A13'}}});sheet.freezePanes.freezeRows(1);sheet.freezePanes.freezeColumns(2);
}
details(selected,t.selected,'SelectedQualityIssues');details(all,t.allrows,'AllCollectionQuality');
wb.recalculate();console.log((await wb.inspect({kind:'table',range:'质量汇总!A26:F26',include:'values,formulas',tableMaxRows:2,tableMaxCols:6,maxChars:1200})).ndjson);console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1000})).ndjson);
for(const [sheetName,range,file]of [['质量汇总','A2:H26','质量汇总预览.png'],['筛选清单','A1:J12','筛选清单预览.png'],['全部采集','A1:J9','全部采集预览.png']]){const img=await wb.render({sheetName,range,scale:1.5,format:'png'});await fs.writeFile(dir+file,new Uint8Array(await img.arrayBuffer()));}
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(dir+'采集质量与去回一致性筛选.xlsx');console.log('WORKBOOK EXPORTED');
