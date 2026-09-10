"""Explain timing availability without turning missing measurements into zeros."""


STATUS_TEXT={
    'committed':'已提交',
    'deadline_expired':'反馈未在截止前提交（计算可能仍在运行）',
    'worker_busy':'本步未启动：上次反馈仍在运行',
    'no_budget':'本步未启动：周期预算已耗尽',
    'snapshot_changed':'状态变化，丢弃',
    'frame_missing':'本步未启动：未获得 ACK 后新图像',
    'operator_abort':'已停止',
}


def ms(info,key):
    value=info.get(key)
    return '—' if value is None else f'{value:.1f} ms'


def computation_text(info):
    if info.get('compute_ms') is not None:return f'计算 {ms(info,"compute_ms")}'
    if info.get('revision_status') in ('no_budget','worker_busy','frame_missing'):
        return '本步未计算'
    if info.get('revision_status')=='deadline_expired':return '计算耗时待后台完成后记录'
    return '计算耗时未记录'


def timing_line(info):
    parts=[f'步骤 {info.get("step","—")}',f'发令间隔 {ms(info,"command_interval_ms")}',
           f'ACK {ms(info,"ack_ms")}',f'ACK 后唤醒 {ms(info,"ack_delivery_ms")}',
           f'等图 {ms(info,"frame_wait_ms")}',
           f'等待反馈 {ms(info,"feedback_wait_ms")} / 可用预算 {ms(info,"feedback_budget_ms")}',
           computation_text(info),STATUS_TEXT.get(info.get('revision_status'),info.get('revision_status',''))]
    return ' | '.join(parts)
