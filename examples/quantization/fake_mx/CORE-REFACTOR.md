# Fake-MX 五算法重构

分支：codex/fake-mx-core-refactor；基线：fake-mx-lite dccd7da36。

## 范围与结构

Linear 仅支持 RTN、FlatQuant、OmniQuant、LHT（配置名 HADAMARD_LEARNING）、RHT，保留原 MXFP4/MXFP8 scheme 名称与核心配置字段。

- quantization/methods/fake_mx.py：显式注册和兼容类名。
- quantization/methods/fake_mx_algorithms/linear.py：唯一 Linear apply、权重 QDQ 和一次性加载入口。
- 同目录 common.py：参数读取、前缀匹配和形状检查。
- flatquant.py、omniquant.py、lht.py、rht.py：各自参数准备与激活变换。
- moe.py：split MoE 适配，注册全部五个 MoE 算法。RTN 直接 QDQ；RHT 载入期用与 Linear 相同的确定性 signs（seed=0）+ normalized FWHT 旋转 w13/w2，FC2 激活变换在 moe_mlp 内用同一 signs；LHT 从 sidecar（lht_params_path）逐专家载入矩阵并做 `W @ inv(T).T` 载入期逆变换（不再要求预变换 checkpoint）；OmniQuant 按 expert-map 载入 per-expert log_scale，权重乘 scale、激活除 scale；FlatQuant 从 sidecar 载入 per-expert FC1/FC2 Kronecker 状态（left/right/diag），载入期逆变换，激活变换在 moe_mlp 内向量化执行。
- quantization/fake_mx.py：沿用原 QDQ 数值；`fake_mx_backend` 配置与 kernels/ 下的可选融合 kernel 适配层（外部 kernel 从未交付，恒走 reference）已移除，`fake_mx_quantize` 直接执行 AMCT 兼容的 reference 实现。

不修改 model runner、模型 patch、通用 MoE dispatch/GMM/通信实现。移除 AutoRound/LWC/LAC 的 scheme、配置白名单、十个示例配置和转换器专属路径。历史任务日志和测评矩阵不改；删除的跟踪文件可从 Git 基线恢复。

## 新算法最短接入步骤

1. 在 fake_mx_algorithms 新增一个模块，继承 FakeMXLinearMethod。
2. 构造时从 self.config 读取一次配置；必要时用 get_weight/get_pertensor_param 声明需要加载的参数。
3. prepare_weight(layer) 负责参数读取与权重变换；不要再执行权重 QDQ 或维护 processed 标志，公共入口统一处理。
4. transform_activation(layer, x) 返回变换后的激活；默认 quantize_activation 负责公共 QDQ，apply 只负责调用它和 F.linear。需要特定裁剪参数的算法（例如 FlatQuant）覆盖 quantize_activation；公共执行器不读取算法专属字段。
5. 在 methods/fake_mx.py 注册两个格式类，在 methods/__init__.py 导出；按现有 modelslim_config.py 的白名单添加量化类型。无需修改模型和执行器。
6. 添加参数缺失/形状检查、配对变换、权重幂等及 FP32/BF16 回归，再做 NPU server smoke 和数据集验证。

算法模块沿用 vLLM scheme 生命周期以减少改动；它们不是可独立运行的训练框架。数学函数可单独测试。TP 分片和融合层参数契约仍沿用基线，未宣称新增支持。

## 数值边界与已知限制

本轮加载工作区时，RHT 已更新为固定 seed 的 Rademacher signs + normalized FWHT。本轮保留该更新，只整理命名与结构。CPU 回归不是 AMCT/Ascend 端到端一致性证明，仍需使用相同版本、尺寸和 seed 核验。

LHT Linear 使用 AMCT 导出的正交 Q，权重和激活配对使用 Q；OmniQuant 保持当前 log_scale 配对缩放，不能宣称实现完整论文全部功能。FlatQuant 保留现有 dtype、矩阵顺序和 clip 行为。统一幂等入口同时避免重复调用时再次切片 FlatQuant 的 TP 矩阵。

性能方面，五算法共享 apply，参数只在 prepare 阶段读取；OmniQuant 前向复用设备上的 scale，不静默退化为 RTN。未引入新 kernel 或更低精度。先验证 eager，再按已有启动脚本的 decode_graph 模式做 NPU A/B。

## 验证命令

```bash
# 仅依赖 CPU PyTorch 和 pytest；框架依赖用隔离进程中的 stub 替代。
python examples/quantization/fake_mx/validation/run_core_cpu_tests.py

# 有完整 vLLM/Ascend 环境时执行真实框架单测。
pytest -q tests/ut/quantization/methods/test_fake_mx_algorithms.py tests/ut/quantization/test_fake_mx_registry.py
```

CPU 检查覆盖五算法 FP32/BF16 权重与激活输出、重复加载、参数错误、注册支持范围及 MoE 权重布局。不能替代真实 checkpoint 加载、TP、MoE dispatch、NPU 数值、图捕获或吞吐测试。Windows 本地缺少 vllm，原仓库 conftest 无法加载；NPU 测评尚未执行。

第二轮清理：算法内部类统一为 FakeMXLinearMethod、FlatQuantLinearMethod、OmniQuantLinearMethod、LHTLinearMethod、RHTLinearMethod；MoE 使用 FakeMXMoEMethod/LHTMoEMethod。长类名只在兼容注册入口保留。注册类只指定 mx_format；算法标识和 checkpoint 契约均位于实现模块。Linear 和 MoE 删除无人使用的 prequantized_weight 开关，Linear 删除无实际约束的权重状态检查。

第三轮清理：LHT 专属矩阵配置、参数分配、准备及 dispatch 前的激活处理归入 LHTMoEMethod。公共 MoE 通过 prepare_weight/quantize_input 扩展点调用，不再判断算法名。专家矩阵仍在原有 dispatch 后的执行位置使用，通用 MoE 通信与 GMM 代码不变。

第四轮收敛（执行链解耦）：per-expert 激活变换抽象为两个通用 transform 槽位（`_fake_mx_fc1_transform` / `_fake_mx_fc2_transform`，签名 `(x, group_list, group_list_type)`），由各算法在 prepare_weight 阶段用闭包绑定自身状态并挂到 layer 上。公共 apply() 只转发这两个槽位加 format/group_size；moe_mlp.py 的 dispatch 后 FC1 变换+QDQ 与 swiglu 后 FC2 变换收敛为两个槽位调用，删除了全部按算法名的分支（原 8 个算法专属参数字段同步移除）。新增 MoE 算法只需实现 transform 工厂，不再修改公共执行与参数传递链；RHT 的 FC1 在 dispatch 前（quantize_input）处理，仅占用 FC2 槽位。

CPU 回归增加 LHT MoE 参数分配、延后激活 QDQ、权重布局及幂等检查；完整框架验证仍需在配置好的 Ascend 环境执行。
