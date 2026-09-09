"""Exercise the actual GUI with virtual valves/camera; never opens real devices."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PyQt5.QtCore import Qt,QPoint,QPointF,QEvent
from PyQt5.QtGui import QMouseEvent,QPainter,QPen,QColor,QFont
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from ..gui.theme import QSS
from ..gui.main_window import ValidationWindow
from ..runtime.hereditary_deployment import transform


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',required=True);parser.add_argument('--out',required=True);parser.add_argument('--ndi',action='store_true');parser.add_argument('--full-occlusion',action='store_true')
    a=parser.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    app=QApplication.instance() or QApplication([]);app.setStyleSheet(QSS);window=ValidationWindow()
    window._save_hardware_config=lambda:None
    window.show();window._set_combo_data(window.hw_profile_preset,'all_mock');window._on_profile_preset_changed(window.hw_profile_preset.currentIndex())
    window.run_root.setText(str(out/'runs'));window.tabs.setCurrentIndex(0);p=window.hereditary_panel
    errors=[];p.fail=lambda error:errors.append(str(error))
    captures=[]
    def wait(predicate,timeout=20):
        end=time.monotonic()+timeout
        while not predicate():
            app.processEvents();time.sleep(.005)
            if errors:raise RuntimeError(errors[-1])
            if time.monotonic()>end:raise TimeoutError('GUI operation timed out')
        app.processEvents()
    def fresh():
        since=time.monotonic()
        wait(lambda:p.frame_provider() is not None and p.frame_provider()[1]>=since)
    def draw(points):
        app.processEvents()
        scale,x,y=p.canvas._layout()
        pixel=lambda point:QPoint(round(x+scale*point[0]),round(y+scale*point[1]))
        QTest.mousePress(p.canvas,Qt.LeftButton,Qt.NoModifier,pixel(points[0]))
        for point in points[1:]:
            event=QMouseEvent(QEvent.MouseMove,QPointF(pixel(point)),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
            QApplication.sendEvent(p.canvas,event)
        QTest.mouseRelease(p.canvas,Qt.LeftButton,Qt.NoModifier,pixel(points[-1]));app.processEvents()
        if errors:raise RuntimeError(errors[-1])
    def capture(name,title,items,description,surface=None):
        surface=surface or window
        app.processEvents()
        surface.grab().save(str(out/(name+'_raw.png')))
        pix=surface.grab();painter=QPainter(pix);painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont('Noto Sans CJK SC',12,QFont.Bold))
        labels=[]
        for number,(widget,label) in enumerate(items,1):
            point=widget.mapTo(surface,QPoint(0,0))
            rect=widget.rect().translated(point).adjusted(-3,-3,3,3)
            painter.setPen(QPen(QColor('#e53935'),3));painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(rect)
            center=QPoint(max(14,rect.left()+10),max(14,rect.top()+9))
            painter.setBrush(QColor('#e53935'));painter.drawEllipse(center,13,13)
            painter.setPen(QColor('white'));painter.drawText(center.x()-7,center.y()-9,14,18,Qt.AlignCenter,str(number))
            labels.append(label)
        painter.end();pix.save(str(out/(name+'.png')))
        captures.append(dict(name=name,title=title,labels=labels,description=description))

    try:
        p.path.setText(str(Path(a.bundle).resolve()));p.load_button.click();wait(lambda:not p.busy)
        if p.runtime is None:raise RuntimeError('model load failed')
        if window.tabs.count()!=4:raise RuntimeError('default workflow must have four stages')
        if p.runtime.initialized or len(p.runtime.history):raise RuntimeError('model load fabricated history')
        p.mapping_button.click()
        capture('01_model','第 1 步 · 模型与紧凑六腔映射',[(p.load_button,'加载仅建立软件模型，不发压力、不记录准备动作'),(p.mapping_button,'六腔分两行选择对应的四个输入')],
                '默认 c0→u0，c1/c2→u1，c3/c4→u2，c5→u3。共享输入的压力联动，各腔限制单独设置。')
        window._connect_valve_group(1);wait(lambda:window.hardware.valve_controller and 1 in window.hardware.valve_controller.connected_groups)
        wait(lambda:not window._valve_connect_thread.isRunning())
        p.chambers.refresh()
        if p.chambers.targets[3].isEnabled():raise RuntimeError('unconnected group was enabled')
        p.runtime.set_mapping([0,1,2,0,2,3]);p.chambers.refresh()
        if p.chambers.targets[0].isEnabled() or p.chambers.targets[2].isEnabled():raise RuntimeError('cross-group shared input was enabled without both groups')
        p.runtime.set_mapping([0,1,1,2,2,3])
        window._connect_valve_group(2);wait(lambda:2 in window.hardware.valve_controller.connected_groups)
        wait(lambda:not window._valve_connect_thread.isRunning())
        window._start_camera();fresh()
        if a.ndi:
            window._connect_ndi();wait(lambda:len(window.hardware.evaluation_samples(0,time.monotonic())['samples'])>0)
        scroll=window.tabs.widget(0);scroll.ensureWidgetVisible(window.hw_conn1_btn)
        capture('02_devices','第 1 步 · 独立连接',[(window.hw_profile_preset,'选择全 Mock 或真机验证'),(window.hw_conn1_btn,'组1独立连接或断开'),(window.hw_conn2_btn,'组2独立连接或断开'),(window.camera_btn,'自动选择 SDK 或 UVC，相机不在运行中切换驱动'),(window.hw_ndi_btn,'NDI 可选连接，未连接也能执行')],
                '连接按钮自动校验配置。低频连接参数在高级设置中。服务器使用动作驱动虚拟相机；真机连接原有 USB 和串口接口。')
        p.chamber_button.click();p.chambers.targets[0].setValue(5);p.chambers.targets[1].setValue(5);p.chambers.targets[3].setValue(5);p.chambers.targets[5].setValue(5)
        p.chambers.start_button.click();wait(lambda:np.all(np.isfinite(p.ack6)) and np.max(abs(p.ack6-5))<.1)
        old_bounds=p.runtime.bounds
        p.chambers.limits[1][1].setValue(4.)
        if p.chambers.apply():raise RuntimeError('limits excluding live pressure were accepted')
        if p.runtime.bounds is not old_bounds:raise RuntimeError('invalid limits partially committed')
        p.chambers.limits[1][1].setValue(150.)
        p.chambers.targets[1].setValue(8.);p.chambers.apply_button.click()
        wait(lambda:abs(p.ack6[1]-8.)<.1 and abs(p.ack6[2]-8.)<.1)
        p.chambers.targets[1].setValue(5.);p.chambers.apply_button.click();wait(lambda:np.max(abs(p.ack6-5))<.1)
        p.chambers.end_button.click();wait(lambda:not p.busy)
        if p.runtime.initialized or len(p.runtime.history):raise RuntimeError('manual preparation created model history')
        capture('03_chambers','全局工具 · 独立手动调压',[(p.chambers.targets[1],'共享输入的目标联动'),(p.chambers.limits[1][1],'每腔 min/max/rise/fall 单独配置，与共享腔及模型取交集'),(p.chambers.apply_button,'应用目标和限制；运行时下一拍生效'),(p.chambers.end_button,'结束调压并保持当前压力；关闭窗口同样停止调压')],
                '无需开始录制。已确认下发栏来自对应阀组 ACK，不是实测气压。连接时不会恢复非零目标。自动执行中点击手动调压会先停止旧计划、等待在途 ACK 再接管。',surface=p.chambers)
        p.chambers.close();window.tabs.setCurrentIndex(1)
        plant=window.hardware.cameras[0]
        fresh();p.auto_button.click();wait(lambda:not p.busy)
        if p.canvas.draft is None:raise RuntimeError('automatic extraction did not produce a draft')
        original=p.canvas.draft.copy()
        scale,x,y=p.canvas._layout();node=7
        point=QPoint(round(x+scale*original[node,0]),round(y+scale*original[node,1]))
        QTest.mousePress(p.canvas,Qt.LeftButton,Qt.NoModifier,point)
        QTest.mouseRelease(p.canvas,Qt.LeftButton,Qt.NoModifier,point+QPoint(2,0));app.processEvents()
        if np.linalg.norm(p.canvas.draft[node]-original[node])<.5:raise RuntimeError('draft node drag failed')
        capture('04_draft','第 2 步 · 自动提取后人工修正',[(p.auto_button,'自动提取完整形状并冻结当前图像'),(p.flip_button,'检查 BASE / TIP，必要时交换'),(p.confirm_button,'拖动黄点完成修正后，确认并计算坐标')],
                '黄色为待确认草稿。拖动黄点调整中心线，包括 base 与 tip；也可以手动重画。初始化应让全臂可见，多个较大分离区域会拒绝自动初始化，不会把残段补成完整测量。')
        p.confirm_button.click();wait(lambda:not p.busy,30)
        if not p.runtime.ready:raise RuntimeError('deployment never became ready')
        if not p.runtime.alignment_confirmed:raise RuntimeError('alignment did not complete')
        capture('05_alignment','第 2 步 · 部署预热通过',[(p.alignment_label,'检查尺度、旋转、base 和形状拟合'),(p.confirm_button,'确认后固定坐标，估计记忆状态并持续观测'),(p.warmup_options,'可展开检查预热时间、连续帧与质量阈值')],
                '当前 5 kPa 初始压力保持不变。首次采用训练一致的当前输入平衡先验，再用完整形状修正 p/h。达到最低时间、连续有效新帧、残差与稳定性阈值才显示模型就绪；不是固定倒计时。')
        window.tabs.setCurrentIndex(2)
        goal=plant.engine.rollout(plant.state,np.tile([.8,.05,.7,.05],(40,1)))[-1]
        p.goal_button.click();draw(transform(goal,plant.matrix));p.horizon.setValue(80)
        p.planning_settings.click()
        capture('06_parameters','第 3 步 · 初始规划参数',[(p.planning_budget,'总计算预算，迭代间检查'),(p.planning_iterations,'每个长度的 B 优化次数'),(p.planning_shooting,'终态优化评估次数'),(p.planning_stride,'搜索步长，达到目标容限即结束')], '修改立即生效并撤销旧计划。初始规划耗时单独显示，不等于在线反馈耗时。',surface=p.planning_dialog)
        p.planning_dialog.hide()
        p.plan_button.click();wait(lambda:not p.busy,40)
        if p.plan is None:raise RuntimeError('planning did not complete')
        if not p.plan['qualified']:raise RuntimeError('automatic horizon search failed: '+str(p.plan['attempts']))
        initial_planning=dict(ms=p.plan['planning_ms'],attempts=p.plan['attempts'],config=p.plan['planning_config'])
        planned_steps=len(p.plan['actions'])
        p.scrubber.setValue(planned_steps//2)
        if p.canvas.preview is None:raise RuntimeError('trajectory scrubber missing')
        if p.plan['actions'].max()<.01:raise RuntimeError('nontrivial goal produced a stationary plan')
        capture('06_target','第 3 步 · 描画目标并规划',[(p.goal_button,'连续画目标中心线，从已对齐 base 到 tip'),(p.display_mode,'选择整体形状叠图或仅骨架'),(p.opacity,'调节透明度，检查与原始相机图像的关系'),(p.plan_button,'按容限自动搜索规划长度'),(p.scrubber,'拖动查看中间预测形状与六腔压力')],
                '红色为目标，青色为模型预测。整体形状由中心线按模型半径扩展，仅用于查看；并非从遮挡图像中恢复的真实轮廓。预览残差是模型内指标。')
        p.display_mode.setCurrentIndex(1)
        capture('07_skeleton','第 3 步 · 骨架显示',[(p.display_mode,'切换显示不改变目标或规划结果')],
                '骨架和整体形状使用同一目标中心线。当前版本不约束障碍物；请使用无接触、无障碍实验场景。')
        p.display_mode.setCurrentIndex(0)
        window.tabs.setCurrentIndex(3)
        p.occlusion.setChecked(True);p.arm_check.setChecked(True)
        wait(lambda:p.frame_provider()[1]>time.monotonic()-.1)
        capture('08_execute','第 4 步 · 开始反馈控制',[(p.occlusion,'真实或虚拟相机均可开启软件遮挡；原图另存'),(p.occlusion_controls.apply_button,'支持多个矩形，设置位置、大小和灰度后应用'),(p.arm_check,'确认当前计划'),(p.execute_button,'正式开始规划动作与反馈控制'),(p.max_skipped,'连续无法及时修正时的停止门限'),(p.feedback_settle,'默认仅等 ACK 后新帧，额外等图会压缩反馈预算'),(p.stop_button,'随时中止并归零')],
                '每步发出压力、收到 ACK、等待新图像，再校正 p/h 并修订剩余动作。无需手画遮挡区域。连续三次没有新帧或没有可信边缘会停止并归零。')
        window.grab().save(str(out/'gui_plan.png'))
        fresh();p.execute_button.click();wait(lambda:not p.busy,30)
        if errors:raise RuntimeError(errors[-1])
        capture('09_results','第 4 步 · 反馈诊断与结束',[(p.feedback_label,'查看计算耗时、可信边缘与证据覆盖'),(p.results,'日志和原始反馈图像保存位置'),(p.stop_button,'正常完成后仍保持末条压力；点击这里释放')],
                '局部证据缺失可能来自遮挡，也可能是光照或预测失配，不等于已准确识别遮挡物。新目标需重新规划。虚拟相机使用同一模型生成，结果只证明软件链路。')
        window.grab().save(str(out/'gui_complete.png'))
        log_path=p.runtime.run_dir/'events.jsonl'
        events=[json.loads(line) for line in log_path.read_text().splitlines()]
        feedback=[e for e in events if e['event']=='feedback']
        completed=[e for e in events if e['event']=='execute_completed']
        command_times=[e['t_command'] for e in events if e['event']=='command_receipt' and e['phase']=='control']
        if not completed or len(feedback)!=planned_steps:raise RuntimeError('missing completion or feedback')
        archive=p.executor.last_execution_dir
        import csv
        archive_meta=json.loads((archive/'metadata.json').read_text())
        if a.ndi and not archive_meta['ndi_samples']:raise RuntimeError('connected NDI not recorded')
        if not a.ndi and archive_meta['ndi_samples']:raise RuntimeError('NDI invented while disconnected')
        with (archive/'commands.csv').open() as handle:command_rows=list(csv.DictReader(handle))
        if len(command_rows)!=planned_steps:raise RuntimeError('missing command CSV rows')
        import cv2
        original=cv2.imread(str(archive/'raw/cam0/00000.png'));used=cv2.imread(str(archive/'frames/00000.png'))
        if original is None or used is None or np.array_equal(original,used):raise RuntimeError('raw and occluded feedback were not separated')
        # A second, short plan is deliberately interrupted through the global
        # tool. It must settle its ACK and relinquish ownership before manual sends.
        p.occlusion.setChecked(False);fresh();window.tabs.setCurrentIndex(2)
        z,u=p.runtime.state_at(time.monotonic());p.target=p.runtime.engine.observe(z,u)
        p.runtime.target_shape=p.target.copy();p.canvas.target=transform(p.target,p.runtime.matrix)
        p.make_plan();wait(lambda:not p.busy,40)
        if not p.plan or not p.plan['qualified']:raise RuntimeError('takeover fixture planning failed')
        matrix=p.runtime.matrix.copy();epoch=p.runtime.history_epoch
        window.tabs.setCurrentIndex(3);p.arm_check.setChecked(True);fresh()
        p.execute_button.click()
        wait(lambda:p.busy and p.runtime.phase=='control')
        p.chambers.show();p.chambers.start_button.click()
        wait(lambda:p.chambers.driving,15)
        wait(lambda:np.max(abs(p.ack6-5))<.1,15)
        p.chambers.close();wait(lambda:not p.busy)
        if p.plan is not None or not p.runtime.initialized:raise RuntimeError('takeover retained plan or lost model history')
        np.testing.assert_array_equal(matrix,p.runtime.matrix)
        if p.runtime.history_epoch!=epoch:raise RuntimeError('manual takeover reset model memory')
        after_events=[json.loads(line) for line in log_path.read_text().splitlines()]
        if not any(e['event']=='execute_held' for e in after_events):raise RuntimeError('automatic owner did not yield to manual hold')
        p.stop_button.click();wait(lambda:not p.busy)
        if max(abs(np.asarray(window.hardware.valve_controller.last_command)))>1e-8:raise RuntimeError('stop did not zero valves')
        blind_verified=False
        if a.full_occlusion:
            errors.clear();p.occlusion.setChecked(False);fresh();window.tabs.setCurrentIndex(2)
            z,u=p.runtime.state_at(time.monotonic());p.target=p.runtime.engine.observe(z,u);p.runtime.target_shape=p.target.copy()
            p.make_plan();wait(lambda:not p.busy,40)
            if not p.plan or not p.plan['qualified']:raise RuntimeError('blind fixture plan failed')
            cells=p.occlusion_controls.table
            while cells.rowCount()>1:cells.removeRow(1)
            for j,v in enumerate([0,0,100,100,35]):cells.cellWidget(0,j).setValue(v)
            p.occlusion.setChecked(True);p.occlusion_controls.apply();window.tabs.setCurrentIndex(3)
            p.arm_check.setChecked(True);fresh();start=time.monotonic();p.execute_button.click()
            while p.busy and time.monotonic()-start<8:app.processEvents();time.sleep(.005)
            if p.busy:raise RuntimeError('complete occlusion did not terminate')
            if not errors or '没有有效图像边缘' not in errors[-1]:raise RuntimeError('full occlusion did not trigger expected stop: '+str(errors))
            errors.clear();folder=p.executor.last_execution_dir
            with (folder/'commands.csv').open() as handle:blind=list(csv.DictReader(handle))
            if len([row for row in blind if row['step']!='zero'])!=3 or blind[-1]['step']!='zero':raise RuntimeError('blind command count/zero incorrect')
            if len(list((folder/'raw/cam0').glob('*.png')))!=3:raise RuntimeError('last failure image missing')
            blind_verified=True;p.occlusion.setChecked(False)
        proposals=[json.loads(file.read_text()) for file in sorted((archive/'feedback_jobs').glob('*.json'))]
        compute_times=[row['wall_ms'] for row in proposals]
        if not compute_times:raise RuntimeError('missing feedback computation artifacts')
        with (archive/'timings.csv').open() as handle:timings=list(csv.DictReader(handle))
        if len(timings)!=planned_steps:raise RuntimeError('missing timing rows')
        summary=dict(compute_completed_jobs=len(compute_times),initial_planning=initial_planning,revision_status_counts={status:sum(e.get('revision_status')==status for e in feedback) for status in set(e.get('revision_status') for e in feedback)},kind='GUI virtual-device integration only',commands=completed[-1]['commands'],feedback_frames=len(feedback),
                     accepted_suffix_updates=sum(e.get('state_committed',False) and e.get('control',{}).get('accepted',False) for e in feedback),
                     compute_ms_percentiles=np.percentile(compute_times,[50,95,100]).tolist(),
                     command_interval_ms_percentiles=np.percentile(np.diff(command_times)*1000,[50,95,100]).tolist(),
                     frame_age_ms_percentiles=np.percentile([e.get('frame_age_ms',0) for e in feedback],[50,95,100]).tolist(),
                     zero_ack=True,ndi_enabled=a.ndi,ndi_samples=archive_meta['ndi_samples'],raw_feedback_separated=True,full_occlusion_stop_verified=blind_verified,alignment_rms_px=p.alignment_error,run_dir=str(p.runtime.run_dir),
                     history_epoch=p.runtime.history_epoch,automatic_horizon=planned_steps,manual_takeover_verified=True,live_limits_verified=True,partial_groups_verified=True,nonzero_initialization_kpa=5.,deployment_ready=p.runtime.ready,initialization_source='camera pixels with manual node edit',
                     visibility_coverage=[e.get('visibility',{}).get('coverage') for e in feedback],
                     physical_hardware_tested=False,physical_control_accuracy=None)
        from .hereditary_guide import write_guide
        write_guide(out,captures,summary)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'COMPLETE').touch();print(json.dumps(summary,indent=2))
    finally:
        window.close();app.processEvents()
if __name__=='__main__':main()
