import fs from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const dest=fileURLToPath(new URL('.',import.meta.url));
const data=JSON.parse(await fs.readFile(dest+'评估结果.json','utf8'));
const inputs=JSON.parse(await fs.readFile(dest+'表格输入.json','utf8'));
const wb=Workbook.create();
const summary=wb.worksheets.add('类别评估');
const detail=wb.worksheets.add('逐次采集审查');
const font='Arial';
for(const sheet of [summary,detail]){
  sheet.showGridLines=false;
  sheet.getRange(sheet===summary?'A1:O36':'A1:N536').format.font={name:font,size:11,color:'#243447'};
  sheet.getRange(sheet===summary?'A1:O36':'A1:N536').format.verticalAlignment='center';
}
summary.tabColor='#244B69';
summary.getRange('A2').values=[['布料数据评估 · 2026-11-18']];
summary.getRange('A2').format.font={size:16,bold:true};
summary.getRange('A3').values=[['采集质量：535个布料CSV；可训练512个（含建议复核12个），不可训练23个（排除10个、数据错误13个）。另10个轨迹测试不纳入。']];
summary.getRange('A4').values=[['分类表现：前20次实验（291CSV）与另一组实验（430CSV）的开发集五折验证；未对535次全量训练，未评估最终留出测试集。']];
summary.getRange('A5').values=[['验证序列数包含去、回两个方向；同一CSV两方向不独立。当前采集数按CSV计。']];
summary.getRange('A3:A5').format.font={size:11,color:'#596C7C'};
const headers=['布料','当前CSV','可训练CSV','不可训练CSV','建议复核','前20实验\nTCN F1','前20实验\nCERA F1','430实验\nTCN F1','430实验\nCERA F1','430实验\n该类CSV数','去回差异率\n中位数','采集幅值\n变异系数','筛选建议','430实验TCN主要错分目标','430实验CERA主要错分目标'];
const target=(s,a)=>s.larger_confusion[a].map(x=>`${x.label}（${x.count}条）`).join('；')||'无';
const rows=data.classes.map(s=>[s.label,s.total,s.eligible,s.total-s.eligible,s.review,s.f1.tcn,s.f1.cera,s.larger_f1.tcn,s.larger_f1.cera,s.larger_csv,s.common_rate_median,s.cv,s.suggestion,target(s,'tcn'),target(s,'cera')]);
summary.getRange('A7:O22').values=[headers,...rows];
summary.tables.add('A7:O22',true,'FabricEvaluation');
summary.getRange('A7:O7').format={fill:'#244B69',font:{name:font,size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:42};
summary.getRange('A8:O22').format.rowHeight=32;
summary.getRange('B8:L22').format.horizontalAlignment='right';
summary.getRange('B8:E23').setNumberFormat('0');
summary.getRange('J8:J23').setNumberFormat('0');
summary.getRange('F8:I22').setNumberFormat('0.0%');
summary.getRange('K8:L22').setNumberFormat('0.0%');
const widths=[24,12,13,14,12,14,14,14,14,15,15,15,21,53,53];
for(let i=0;i<widths.length;i++)summary.getRangeByIndexes(6,i,17,1).format.columnWidth=widths[i];
summary.getRange('A23').values=[['合计']];
for(const col of ['B','C','D','E'])summary.getRange(col+'23').formulas=[[`=SUM(${col}8:${col}22)`]];
summary.getRange('J23').formulas=[['=SUM(J8:J22)']];
summary.getRange('A23:E23').format.font={bold:true};
summary.getRange('A23:O23').format.borders={top:{style:'thin',color:'#AABAC6'}};
const notes=[
 ['A25','筛选参考：两实验×两模型全部F1≥85% → 优先保留；全部＜80% → 优先暂不纳入；仅430实验两模型均＜80% → 重点复核；其余待验证。'],
 ['A26','参考线仅用于本次类别讨论；“暂不纳入”不表示采集异常。模型错分的单条数据不据此排除。F1兼顾漏分和误分，越高越好。'],
 ['A27','去回差异与幅值变异系数：使用全部可训练记录、未标准化|ΔBy|；后者反映同类各CSV代表幅值的波动，不能代替分类结果。'],
 ['A28','错分条数：该真实类别被错分为目标类别的方向序列数；每个模型分别统计，不把两模型预测相加为独立样本。'],
 ['A29','审查规则：'+data.scope.review_rule+'；所有原始数据、旧模型和实验选择保持原样；排除详情见逐次采集审查。'],
 ['A30','仿皮毛一体：新增13次回程采样缺口超过20ms，本次不准入；应复采这些记录，类别仍建议保留。'],
 ['A31','数据源：'+data.scope.source],
 ['A32','前20实验源：'+data.scope.model_source+'/metadata.json'],
 ['A33','430实验源：'+data.larger_model.task+'/metadata.json'],
 ['A34','评价限制：430实验各类数量不均，两实验数据重叠，不能当独立复验；批次跨训练与验证，不能证明新批次识别率。'],
 ['A35','验证范围：前20实验464条方向序列/232CSV；430实验686条方向序列/343CSV。改变候选类别后需重新训练、用新独立采集验证。']
];
for(const [cell,text]of notes)summary.getRange(cell).values=[[text]];
summary.getRange('A25:A35').format.font={size:11,color:'#596C7C'};
summary.getRange('A25:O35').format.rowHeight=25;
summary.freezePanes.freezeRows(7);summary.freezePanes.freezeColumns(1);
summary.getRange('M8:M22').conditionalFormats.add('containsText',{text:'暂不纳入',format:{fill:'#FCEAE7',font:{color:'#A5362D'}}});
summary.getRange('M8:M22').conditionalFormats.add('containsText',{text:'重点复核',format:{fill:'#FFF3D8',font:{color:'#7F5B10'}}});
summary.getRange('M8:M22').conditionalFormats.add('containsText',{text:'优先保留',format:{fill:'#E8F1EE',font:{color:'#28614E'}}});
detail.getRange('A1:N536').values=[inputs.detailheaders,...inputs.details];
detail.tables.add('A1:N536',true,'CaptureReview');
detail.getRange('A1:N1').format={fill:'#244B69',font:{name:font,size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,rowHeight:42};
detail.getRange('A2:N536').format.rowHeight=25;
const detailwidths=[24,33,10,22,12,18,18,22,20,28,100,72,72,140];
for(let i=0;i<detailwidths.length;i++)detail.getRangeByIndexes(0,i,536,1).format.columnWidth=detailwidths[i];
detail.getRange('F2:G536').setNumberFormat('0.00%');
detail.getRange('I2:I536').setNumberFormat('0.0%');
detail.getRange('H2:H536').setNumberFormat('0.00');
detail.getRange('J2:J536').setNumberFormat('0.00');
detail.getRange('D2:D536').conditionalFormats.add('containsText',{text:'本实验排除',format:{fill:'#FCEAE7',font:{color:'#A5362D'}}});
detail.getRange('D2:D536').conditionalFormats.add('containsText',{text:'数据错误',format:{fill:'#FCEAE7',font:{color:'#A5362D'}}});
detail.freezePanes.freezeRows(1);detail.freezePanes.freezeColumns(2);
wb.recalculate();
const check=await wb.inspect({kind:'table',range:'类别评估!A23:E23',include:'values,formulas',tableMaxRows:2,tableMaxCols:5,maxChars:1500});
console.log(check.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000});
console.log(errors.ndjson);
for(const [name,range,file] of [['类别评估','A2:M23','类别评估预览.png'],['逐次采集审查','A1:J12','采集明细预览.png']]){
 const image=await wb.render({sheetName:name,range,scale:1.5,format:'png'});
 await fs.writeFile(dest+file,new Uint8Array(await image.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(dest+'布料数据评估统计.xlsx');
console.log('EXPORTED',dest+'布料数据评估统计.xlsx');
