"""Lightweight bounded repaint of received signal and class probabilities."""
import math
from PyQt5 import QtWidgets as W,QtGui as G,QtCore as C


class EarlyTimeline(W.QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.setMinimumHeight(235);self.points=[];self.predictions=[];self.labels=[];self.completion=None;self.origin=None;self.cursor=None
    def clear(self):
        self.points=[];self.predictions=[];self.completion=None;self.origin=None;self.cursor=None;self.update()
    def append_points(self,points):
        self.points.extend(points)
        if len(self.points)>12000:self.points=self.points[::2]
        if self.origin is None and self.points:self.origin=self.points[0][0]
        self.update()
    def append_prediction(self,event):
        self.predictions.append(event)
        if len(self.predictions)>1000:self.predictions=self.predictions[-1000:]
        if event.get('completion'):self.completion=event['completion']
        self.update()
    def load(self,trace):
        self.points=list(trace.get('points',[]));self.predictions=list(trace.get('predictions',[]));self.labels=list(trace.get('labels',[]));self.completion=trace.get('completion');self.origin=self.points[0][0] if self.points else None;self.cursor=None;self.update()
    def mousePressEvent(self,event):
        if self.points:
            span=max(1.,self.points[-1][0]-self.points[0][0]);self.cursor=max(0,min(span,(event.pos().x()-48)/max(1,self.width()-65)*span));self.update()
    def paintEvent(self,event):
        p=G.QPainter(self);p.setRenderHint(G.QPainter.Antialiasing);p.fillRect(self.rect(),G.QColor('#141d2b'))
        w=self.width();h=self.height();left=48;right=max(left+1,w-17);top=22;middle=h*.45;bottom=h-32
        origin=self.origin if self.origin is not None else (self.predictions[0]['t_s'] if self.predictions else 0)
        span=max(1.,max([v[0]-origin for v in self.points]+[e['t_s']-origin for e in self.predictions]+[1.]))
        def xx(t):return left+(t-origin)/span*(right-left)
        p.setPen(G.QColor('#96a6bf'));p.drawText(8,16,'|dBy|');p.drawText(8,int(middle+20),'概率')
        for y in (middle-5,bottom):p.setPen(G.QPen(G.QColor('#34445a'),1));p.drawLine(left,int(y),right,int(y))
        p.setPen(G.QColor('#96a6bf'))
        for i in range(5):
            x=left+(right-left)*i/4;p.drawText(int(x)-10,h-12,f'{span*i/4:.1f}')
        p.drawText(int(w*.40),h-1,'有效扫描后 / 秒（点击查看时刻）')
        if not self.points and not self.predictions:
            p.drawText(self.rect(),C.Qt.AlignCenter,'等待数据 · 收满有效历史后开始判断');return
        completion_t=self.completion.get('t_s') if self.completion else None
        points=self.points[::max(1,len(self.points)//max(100,w))]
        amplitude=max([abs(v[2]) for v in points]+[1.]);previous=None
        for t,x,value in points:
            pos=C.QPointF(xx(t),middle-12-(value/amplitude)*(middle-top-20))
            p.setPen(G.QPen(G.QColor('#738093' if completion_t is not None and t>completion_t else '#82afff'),1.1))
            if previous is not None:p.drawLine(previous,pos)
            previous=pos
        plot_top=middle+28;plot_h=max(1,bottom-plot_top)
        p.setPen(G.QPen(G.QColor('#5c6e86'),1,C.Qt.DotLine));p.drawLine(left,int(plot_top+.05*plot_h),right,int(plot_top+.05*plot_h))
        if self.predictions:
            scores=self.predictions[-1].get('probabilities',[]);leaders=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:3]
            for order,index in enumerate(leaders):
                color=G.QColor.fromHsv((index*137+210)%360,130,235);p.setPen(G.QPen(color,1.7));previous=None
                for e in self.predictions:
                    if index>=len(e.get('probabilities',[])):continue
                    pos=C.QPointF(xx(e['t_s']),plot_top+(1-e['probabilities'][index])*plot_h)
                    if previous is not None:p.drawLine(previous,pos)
                    p.drawEllipse(pos,2.3,2.3);previous=pos
                name=self.labels[index] if index<len(self.labels) else str(index)
                p.drawText(left+order*max(1,(right-left)//3),int(middle+18),p.fontMetrics().elidedText(name,C.Qt.ElideRight,max(30,(right-left)//3-8)))
        if completion_t is not None:
            p.setPen(G.QPen(G.QColor('#66d9a2'),2));x=xx(completion_t);p.drawLine(int(x),top,int(x),int(bottom));p.drawText(min(int(x)+5,right-85),top+12,'锁定 / 已完成')
        if self.cursor is not None:
            p.setPen(G.QPen(G.QColor('#e7b777'),1,C.Qt.DashLine));x=xx(origin+self.cursor);p.drawLine(int(x),top,int(x),int(bottom));p.drawText(left+5,top+12,f'查看 {self.cursor:.2f}秒')
