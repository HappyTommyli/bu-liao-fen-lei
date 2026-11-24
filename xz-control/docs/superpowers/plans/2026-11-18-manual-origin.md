# Manual origin Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
**Goal:** 人工建立固定机械原点，保存并安全返回采集起点。
**Architecture:** AxisConfig扩展起点，plan_steps展开起点路径，Worker跟踪本次人工标定有效性，UI提供人工对位/确认/起点管理并持久化。
**Tech Stack:** Python/PySide6/unittest；兼容既有固件。
**Spec:** ../specs/2026-11-18-manual-origin.md
## Global Constraints
不动实物、不写固件或数据；CSV23列不变；重启不恢复可信坐标。
## Review Focus
ZERO仅ACK不能作为完成；临时对位不能启动自动动作；换算或基准名变化不能复用旧起点；逐段边界检查；设置与函数旧版兼容。
## Tasks
- [x] 模型TDD：起点路线三段、无起点/不安全高度拒绝、跨轮验证/函数导入导出兼容。
- [x] 后台TDD：确认原点只在真实零状态后有效；临时对位不能运行函数；断开/比例改变/单轴设零失效；元数据。
- [x] 界面TDD：确认弹窗，当前坐标存起点，起点/名称重启恢复但有效性不恢复，动作编辑和临时模式禁用自动运行。
- [x] 全套测试/独立审查/截图检查后选择性同步到用户目录，原目录再回归。
## Ledger
- 用户明确选择人工标定并授权脚本修改；沿用已建立的隔离工作树，保留原文件备份后选择性交付。

- Baseline: isolated snapshot of current user app; all 159 tests passed.
- Model RED→GREEN: capture_start three-leg routing, higher initial Z, missing profile, unsafe clearance/intermediate bounds, legacy config compatibility, JSON round-trip.
- Backend RED→GREEN: temporary mode blocks automation, explicit confirmation requires ACK and fresh zero status, reconnect/scale/partial-zero invalidation, metadata.
- UI RED→GREEN: confirmation, current-position capture, point persistence without restored trust, temp-mode lockout and signature matching.
- Initial full suite: 171 tests passed; preparation fixture now uses actual simulated confirmation.
- Independent review manual_origin_review: one Important issue, no Critical/minor findings. Operator alignment accuracy and physical collision clearance remain hardware verification responsibilities.
- Review fix RED→GREEN: applied ppmm round trip permanently clears saved point and emits start_cleared.
- Additional fix RED→GREEN: fault while awaiting zero status cancels pending confirmation.
- Startup regression RED→GREEN: restore saved parameters without coordinate trust before first configure; saved nondefault ppmm does not discard retained point.
- Final candidate suite 175/175 passed in 24.682s. Qt/Worker DEMO integration completed 2 independent fixed-start captures; final (15,5) pulses; reopening preserved point, required new confirmation and generated new calibration ID. serial.Serial guarded against any real port open.
- Screenshots inspected; CSV23 columns and all firmware unchanged. 13 files selectively delivered after hash checks; backups retained in isolated delivery-backup. No git merge/push/commit requested or performed.
