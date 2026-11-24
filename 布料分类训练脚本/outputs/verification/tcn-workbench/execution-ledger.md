# SDD ledger — plan: docs/superpowers/plans/2026-11-19-tcn-workbench.md
Ruling: 用户已批准两个设计并要求开始；直接在当前feature分支codex/cloth-gpu-standard-pipeline修改，保持桌面启动路径与本地环境，未创建额外worktree或触及main。
Ruling: inline执行四项任务，末尾fresh reviewer整体审查；无需重复请求方案批准。
Pre-flight: Task1 candidate/spec接口供Task2 fit/evaluate与Task3冻结加载使用；统一可选candidate字典，不修改旧配置字典。
Pre-flight: Task3加载模型时恢复model.candidate，Task2 evaluate可从模型读取，Task4从配置生成同一候选，无重复切片实现。
Task1: RED observed 13 failures (new modules absent); baseline regression launched before edits.
Task 1: complete (commits e701093..702621b, tests: .venv/bin/python -m pytest -q tests/test_tcn_study.py → 13 passed in 2.95s)
Task 2: complete (commits 702621b..f64b962, tests: .venv/bin/python -m pytest -q tests/test_tcn_training.py tests/test_modern_training.py → 19 passed in 4.91s)
Task 3: complete (commits f64b962..d8399ef, tests: .venv/bin/python -m pytest -q tests/test_tcn_engine.py tests/test_modern_engine.py tests/test_recognition.py → 5 passed in 16.25s)
Task 4: Ruling: 默认入口使用TCNTrainerWindow子类，旧TrainerWindow保持4页，复用原数据/成果控制器，避免破坏现有CLI和旧UI测试。
Task 4: Ruling: 1024×640下原拟定上下布局使参数viewport仅62像素；实测后改为训练区至少740逻辑像素时使用双栏，结果/数据页仍响应式切换。12种尺寸/缩放组合检查全部通过。
Task 4: Ruling: 识别无需修改engine.py：加载时在model.candidate保存已验证候选，现有modern evaluate入口读取该属性，验证/测试/识别共用同一聚合实现。
Task 4: TDD RED observed missing focused UI (8 failures); parameter widget test corrected to QComboBox.setCurrentIndex as designed. Many-class plot RED observed minimum height170; added adaptive height for label readability.
Verification: baseline187 passed; interim full214 passed; actual RTX3060 GPU four candidates FP32 forward/backward+optimizer memory probes passed, microbatch16 for all; no real cloth training launched.
Task 4: Ruling: 原生X11启动截图发现resizeEvent读取旧子控件宽度，初次显示误切上下布局；增加先失败的首次show测试，改为父窗口拥有的0ms定时器在布局后重算，10项专用UI测试通过，原生xcb复测双栏orientation=1。
Verification: 原生X11/xcb可用区1850×1053，默认1420×900窗口已实际显示与截图；并非仅离屏验证。Wayland未运行，本机当前X11会话。
Task 4: complete (commits d8399ef..83797eb, tests: env QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q --tb=short → 216 passed in 92.48s (0:01:32))
Final review: fresh reviewer gpt-6.1-terra/high on e701093..83797eb; no Critical, 2 Important (window microbatch cap; update-axis plot/export), 1 Minor (actual window preview/detail).
Review ruling: accepted both Important after code verification. New TCN microbatch allowed0–128 independently; legacy cap unchanged. Added failing parent2/window-micro8 test, then CPU+actual CUDA training tests passed.
Review ruling: added recorded optimizer_updates axis with legacy epoch fallback and live/result PNG export; failing axis test then passed. Cleared stale plot history on empty model state (RED->GREEN).
Review ruling: Minor also implemented: actual parent/window counts, durations and per-window result table. Added failing preview/table tests then passed. Reviewer did not independently rerun environment tests; no second reviewer dispatched.
