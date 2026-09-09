from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QCheckBox,QTableWidget,QDoubleSpinBox
from ..perception.software_occlusion import OcclusionConfig

class OcclusionControls(QWidget):
    def __init__(self,parent):
        super().__init__(parent);self.config=OcclusionConfig();root=QVBoxLayout(self);root.setContentsMargins(0,0,0,0)
        self.enabled=QCheckBox('软件遮挡测试（真实 / 虚拟相机均可）');root.addWidget(self.enabled)
        self.table=QTableWidget(0,5);self.table.setHorizontalHeaderLabels(['x %','y %','宽 %','高 %','灰度']);self.table.setMaximumHeight(125)
        self.table.horizontalHeader().setStretchLastSection(True)
        for i in range(5):self.table.setColumnWidth(i,86)
        root.addWidget(self.table);self.add_region()
        row=QHBoxLayout();root.addLayout(row)
        self.add_button=QPushButton('添加区域');self.remove_button=QPushButton('删除选中');self.apply_button=QPushButton('应用遮挡参数')
        self.add_button.clicked.connect(lambda:self.add_region((20,20,10,10,35)))
        self.remove_button.clicked.connect(lambda:self.table.removeRow(self.table.currentRow()) if self.table.currentRow()>=0 else None)
        self.apply_button.clicked.connect(self.apply)
        for w in (self.add_button,self.remove_button,self.apply_button):row.addWidget(w)
        self.status=QLabel('图像坐标百分比；只覆盖反馈图像，原图另存。初始化时关闭。');self.status.setWordWrap(True);root.addWidget(self.status)
        self.enabled.toggled.connect(self.apply)
    def add_region(self,values=(46.1,46.25,8.75,11.67,35)):
        if self.table.rowCount()>=16:return
        i=self.table.rowCount();self.table.insertRow(i)
        for j,v in enumerate(values):
            box=QDoubleSpinBox();box.setDecimals(2);box.setRange(0,255 if j==4 else 100);box.setValue(v);self.table.setCellWidget(i,j,box)
    def apply(self):
        try:
            rows=tuple(tuple(self.table.cellWidget(i,j).value() for j in range(5)) for i in range(self.table.rowCount()))
            self.config=OcclusionConfig(self.enabled.isChecked(),rows)
            self.status.setText(f'已应用 {len(rows)} 个区域；原图和反馈图分开保存。')
        except ValueError as error:
            self.enabled.blockSignals(True);self.enabled.setChecked(self.config.enabled);self.enabled.blockSignals(False)
            self.status.setText(str(error))
