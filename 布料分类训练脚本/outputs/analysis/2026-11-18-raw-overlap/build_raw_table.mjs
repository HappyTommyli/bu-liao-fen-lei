import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const dir=fileURLToPath(new URL('.',import.meta.url));
const t=JSON.parse(await fs.readFile(dir+'表格输入.json','utf8'));
const d=JSON.parse(await fs.readFile(dir+'重合度分析.json','utf8'));
const wb=Workbook.create();const summary=wb.worksheets.add('布料去留');const pairs=wb.worksheets.add('两两重合');const records=wb.worksheets.add('采集明细');
const headerStyle={fill:'#244B69',font:{name:'Arial',size:11,color:'#FFFFFF',bold:true},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:46};
function base(sheet,last,columns){sheet.showGridLines=false;sheet.getRangeByIndexes(0,0,last,columns).format.font={name:'Arial',size:11,color:'#243447'};sheet.getRangeByIndexes(0,0,last,columns).format.verticalAlignment='center';}
base(summary,36,11);base(pairs,106,17);base(records,t.recordrows.length+1,13);summary.tabColor='#244B69';
summary.getRange('A2').values=[['按原始磁数据选择布料']];summary.getRange('A2').format.font={size:16,bold:true};
summary.getRange('A3').values=[[`本次固定扫描${d.scope.csv_count}个布料CSV，${d.scope.eligible}个参与比较；10个排除、25个数据错误、1个来源变化暂不评估。`]];
summary.getRange('A4').values=[['仅使用原始|ΔBy|。分别比较去回程2mm位置箱，不做逐条标准化，不使用任何训练分数或预测。']];
summary.getRange('A5').values=[['去留按整套保留组合判断；“最相似布料”来自完整15类，可能已在该方案中移出。']];
summary.getRange('A7:K22').values=[t.headers,...t.rows];summary.tables.add('A7:K22',true,'RawFabricSelection');summary.getRange('A7:K7').format=headerStyle;
summary.getRange('A8:K22').format.rowHeight=32;summary.getRange('B8:E22').format.horizontalAlignment='right';
summary.getRange('B8:C23').setNumberFormat('0');summary.getRange('D8:D22').setNumberFormat('0.0');summary.getRange('E8:E22').setNumberFormat('0.0%');summary.getRange('G8:G22').setNumberFormat('0.0%');summary.getRange('H8:H22').setNumberFormat('0.00');
const widths=[24,13,13,22,18,24,24,20,20,23,100];widths.forEach((w,i)=>summary.getRangeByIndexes(6,i,17,1).format.columnWidth=w);
summary.getRange('A23').values=[['合计']];summary.getRange('B23:C23').formulas=[['=SUM(B8:B22)','=SUM(C8:C22)']];summary.getRange('A23:C23').format.font={bold:true};
summary.getRange('I8:J22').conditionalFormats.add('containsText',{text:'换布候选',format:{fill:'#FFF1D9',font:{color:'#805A13'}}});
summary.getRange('I8:J22').conditionalFormats.add('cellIs',{operator:'equal',formula:'"保留"',format:{fill:'#E8F1EE',font:{color:'#28614E'}}});
const notes=[
 ['A25','重合率：两类第5～95百分位波动带的交集面积/并集面积；越低越分开，不是样本重复比例或错分率。'],
 ['A26','距散比：两个类中位波形RMS差 / 两类各自重复采集RMS波动之和；越高越分开。'],
 ['A27',`严格方案：两两重合≤20%、距散比≥2；保留${d.selection.strict.labels.join('、')}。实际最高重合${(100*d.selection.strict.max_overlap).toFixed(1)}%。`],
 ['A28',`五类折中方案：两两重合≤50%、距散比≥1；保留${d.selection.moderate.labels.join('、')}。实际最高重合${(100*d.selection.moderate.max_overlap).toFixed(1)}%。`],
 ['A29','方案枚举全部组合，先取类别数最多者，再取同类幅值CV总和最小者。其他同规模组合也可能可行；阈值是探索性工程选择。'],
 ['A30','严格方案针对主要靠幅值/粗波形就明显区分的目标。多数高重合配对仍有纹理差异，换布候选不表示材料无法分类。'],
 ['A31','同编号段的重合明显低于全量汇总：跨段幅值漂移会扩大重合。编号段不等于已验证的采集批次。'],
 ['A32','同名重新采集按CSV和metadata指纹重新审查；未完成/读取中变化暂不评估；分析后的新增不自动更新此报告。'],
 ['A34','数据源：'+d.scope.source],
 ['A35','审查规则：'+d.scope.rule_version+'。未修改原始数据、布料标签或训练选择。完整定义见随附分析报告。']
];for(const [cell,value]of notes)summary.getRange(cell).values=[[value]];
summary.getRange('A25:A35').format.font={size:11,color:'#596C7C'};summary.getRange('A25:K35').format.rowHeight=25;summary.freezePanes.freezeRows(7);summary.freezePanes.freezeColumns(1);
function table(sheet,headers,rows,name,widths){const nr=rows.length+1;sheet.getRangeByIndexes(0,0,nr,headers.length).values=[headers,...rows];const last=String.fromCharCode(64+headers.length);sheet.tables.add(`A1:${last}${nr}`,true,name);sheet.getRangeByIndexes(0,0,1,headers.length).format=headerStyle;sheet.getRangeByIndexes(1,0,rows.length,headers.length).format.rowHeight=27;for(let i=0;i<widths.length;i++)sheet.getRangeByIndexes(0,i,nr,1).format.columnWidth=widths[i];sheet.freezePanes.freezeRows(1);sheet.freezePanes.freezeColumns(2);}
table(pairs,t.pairheaders,t.pairrows,'RawPairOverlap',[24,24,21,23,22,22,17,20,20,25,25,27,28,22,24,27,32]);
for(const col of ['C','H','I','J','K','L','M','N','P'])pairs.getRange(`${col}2:${col}106`).setNumberFormat('0.0%');pairs.getRange('D2:G106').setNumberFormat('0.00');
table(records,t.recordheaders,t.recordrows,'RawCaptureReview',[24,35,22,25,24,24,25,25,26,85,72,72,140]);records.getRange(`E2:I${t.recordrows.length+1}`).setNumberFormat('0.00');
records.getRange(`C2:C${t.recordrows.length+1}`).conditionalFormats.add('containsText',{text:'data_error',format:{fill:'#FCEAE7',font:{color:'#A5362D'}}});
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'布料去留!A23:C23',include:'values,formulas',tableMaxRows:2,tableMaxCols:3,maxChars:900})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:900})).ndjson);
for(const [sheetName,range,file]of [['布料去留','A2:J23','统计表预览.png'],['两两重合','A1:I9','配对明细预览.png'],['采集明细','A1:I10','采集明细预览.png']]){const blob=await wb.render({sheetName,range,scale:1.5,format:'png'});await fs.writeFile(dir+file,new Uint8Array(await blob.arrayBuffer()));}
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(dir+'原始数据布料去留统计.xlsx');console.log('EXPORTED RAW-ONLY WORKBOOK');
