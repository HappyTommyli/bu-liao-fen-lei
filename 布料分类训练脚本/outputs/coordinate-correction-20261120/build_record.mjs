import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const dir=fileURLToPath(new URL('.',import.meta.url));
const data=JSON.parse(await fs.readFile(dir+'workbook-input.json','utf8'));
const wb=Workbook.create();
const changed=wb.worksheets.add('修改记录');
const all=wb.worksheets.add('全部检查');
const header={fill:'#25475F',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeight:46};
function initialize(sheet,rows,cols){
  sheet.showGridLines=false;
  const range=sheet.getRangeByIndexes(0,0,rows,cols);
  range.format.font={name:'Arial',size:11,color:'#23354A'};
  range.format.verticalAlignment='center';range.format.rowHeight=28;
}
function columns(sheet,widths,rows){widths.forEach((width,i)=>sheet.getRangeByIndexes(0,i,rows,1).format.columnWidth=width);}
const lastChanged=8+data.modified.length,lastAll=10+data.allrecords.length;
initialize(changed,lastChanged,21);initialize(all,lastAll,12);
changed.tabColor='#25475F';
changed.getRange('A2').values=[['X坐标平移修改记录']];changed.getRange('A2').format.font={size:16,bold:true};
changed.getRange('A3').values=[['2026-11-20：整体右移24.348 mm（6087脉冲），同时更新X位置和X目标位置。']];
changed.getRange('A4').values=[['已修改CSV数']];changed.getRange('B4').formulas=[["='全部检查'!C7"]];
changed.getRange('D4').values=[['未修改CSV数']];changed.getRange('E4').formulas=[["='全部检查'!D7"]];
changed.getRange('G4').values=[['原始内容唯一数']];changed.getRange('H4').values=[[data.unique]];
changed.getRange('A5').values=[['复查范围：去程51–140 mm、回程30–120 mm。仍有数据错误的记录，具体原因见右侧复查列。']];
changed.getRange('A6').values=[['磁场、时间戳、有效标记等其他CSV字段逐行核对未变；原始文件和metadata均已完整备份。']];
changed.getRangeByIndexes(7,0,data.modified.length+1,21).values=[data.headers,...data.modified];
changed.tables.add(`A8:U${lastChanged}`,true,'CoordinateChanges');
changed.getRange('A8:U8').format=header;
columns(changed,[21,22,28,14,14,14,14,14,16,16,22,66,145,155,72,72,20,16,18,72,72],lastChanged);
changed.getRange(`D9:J${lastChanged}`).setNumberFormat('0.000');
changed.getRange(`R9:S${lastChanged}`).setNumberFormat('#,##0');
changed.getRange('B4').setNumberFormat('#,##0');changed.getRange('E4').setNumberFormat('#,##0');changed.getRange('H4').setNumberFormat('#,##0');
changed.getRange(`K9:K${lastChanged}`).conditionalFormats.add('containsText',{text:'data_error',format:{fill:'#FBE6E5',font:{color:'#9E2D31'}}});
changed.getRange(`K9:K${lastChanged}`).conditionalFormats.add('containsText',{text:'excluded',format:{fill:'#FFF1D3',font:{color:'#805A16'}}});
changed.freezePanes.freezeRows(8);changed.freezePanes.freezeColumns(3);

all.getRange('A2').values=[['全部原始CSV检查范围']];all.getRange('A2').format.font={size:16,bold:true};
all.getRange('A3').values=[['只修改基线恒定−24.348 mm且去程起点一致的已完成记录；其他起点及需人工核对的文件未修改。']];
all.getRange('A4:D7').values=[['目录','扫描CSV数','已修改CSV数','未修改CSV数'],['布料采集数据',null,null,null],['布料采集数据新',null,null,null],['合计',null,null,null]];
all.getRange('A4:D4').format=header;
all.getRangeByIndexes(9,0,data.allrecords.length+1,12).values=[data.allheaders,...data.allrecords];
all.tables.add(`A10:L${lastAll}`,true,'AllCoordinatesReviewed');all.getRange('A10:L10').format=header;
for(const row of [5,6]){
  all.getRange(`B${row}`).formulas=[[`=COUNTIFS(A11:A${lastAll},A${row})`]];
  all.getRange(`C${row}`).formulas=[[`=COUNTIFS(A11:A${lastAll},A${row},D11:D${lastAll},"已平移")`]];
  all.getRange(`D${row}`).formulas=[[`=B${row}-C${row}`]];
}
for(const col of ['B','C','D'])all.getRange(`${col}7`).formulas=[[`=SUM(${col}5:${col}6)`]];
all.getRange('A7:D7').format.font={bold:true};all.getRange('B5:D7').setNumberFormat('#,##0');
columns(all,[23,22,28,14,52,16,16,16,16,16,145,155],lastAll);
all.getRange(`F11:J${lastAll}`).setNumberFormat('0.000');all.freezePanes.freezeRows(10);all.freezePanes.freezeColumns(3);

wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'全部检查!A4:D7',include:'values,formulas',tableMaxRows:4,tableMaxCols:4,maxChars:1800})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
for(const [sheetName,range,file] of [['修改记录','A2:K16','修改记录预览.png'],['全部检查','A2:H17','全部检查预览.png']]){
  const image=await wb.render({sheetName,range,scale:1.4,format:'png'});await fs.writeFile(dir+file,new Uint8Array(await image.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(dir+'坐标平移修改记录.xlsx');
console.log('EXPORTED',dir+'坐标平移修改记录.xlsx');
