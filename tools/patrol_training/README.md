# 大棚巡检语言微调使用手册

训练针对“小车听懂人怎么指挥巡检”。例如“先往左拐，到了第一排草莓进去看两趟，最后在右门前五米停”，模型需要提取立即左转、第一排草莓、两趟、右门和五米，并保留动作顺序。程序从原话核对数字和单位。每个行口、作物、门和停车位置都由运行时观测来确定。

## 1. 先找到文件

Windows 总目录为 `D:\Qwen3-VL-2B\patrol_training`。训练使用 `E:\ANACONDA\envs\unet\python.exe` 和 RTX 4060 Laptop GPU。训练用的附加 Python 库装在总目录内。

| 位置（相对总目录） | 用途 |
| --- | --- |
| `Start-Patrol-LoRA.cmd` | 启动供 Ubuntu 调用的 Windows 模型和原有的日志接收端 |
| `adapter\patrol-lora.gguf` | 供现有 Q4 推理模型加载的巡检适配器 |
| `adapter\best.pt`、`adapter\history.json` | 最优轮的两个训练矩阵和各轮损失 |
| `base\` | 用于训练的官方 BF16 权重、分词器和版本记录 |
| `data\train.jsonl`、`validation.jsonl`、`test.jsonl` | 训练、选轮次和最终测试用的独立指令 |
| `evaluation\` | 原版和微调版实际回答、期望步骤、错误与汇总 |
| `scripts\` | 下载、生成数据、训练、评测脚本，在 Git 的 `tools/patrol_training` 有副本 |
| `wheels\`、`deps\` | 附加依赖的离线安装包和已安装位置 |
| `activation_cache\`、`runs\` | 可重新生成的训练中间特征，以及各次训练保留的权重 |

## 2. 用训练好的模型跑仿真

1. 关闭之前占用 18080 的模型窗口。在 Windows 双击 `D:\Qwen3-VL-2B\patrol_training\Start-Patrol-LoRA.cmd`，等窗口显示模型加载完成。这个启动器同时加载原版视觉投影器和巡检适配器。适配器默认强度为零，大棚节点只在解析巡检指令时为该请求开启它。
2. 在 Ubuntu 终端一更新并启动大棚。

   ```bash
   cd ~/robot_ws
   git switch codex/two-room-mapping
   git pull --ff-only
   source /opt/ros/humble/setup.bash
   colcon build --packages-select camera_processing --symlink-install
   source install/setup.bash
   ros2 launch camera_processing two_room_mapping.launch.py world:=greenhouse_routes.sdf vlm_server:=http://192.168.19.1:18080 patrol_lora:=true
   ```

3. 在 Ubuntu 终端二打开指令输入，等画面和地图刷新后输入指令。

   ```bash
   cd ~/robot_ws
   source install/setup.bash
   ros2 run camera_processing route_console
   ```

   可以先试“以0.3米/秒的速度行驶，先左拐，遇到第一条作物行就进去巡一趟，走不通就退回来，再从右门出去”。要只看模型解析，在终端二运行下面这一行，小车不会收到运动任务。

   ```bash
   GREENHOUSE_PATROL_LORA=true ros2 run camera_processing route_console --server http://192.168.19.1:18080 --preview '先左拐，遇到第一条作物行就进去巡两趟，在右门前两米停下'
   ```

4. 要对比原模型，可在同一 Windows 服务上用 `patrol_lora:=false` 重启大棚节点。原有 Windows 启动器也可配合这个选项使用。

启用训练版时，当前默认强度为 **0.75**。这个数表示将学到的权重改变量乘以 0.75。以后重新训练，可以用验证集选新值，并在 launch 命令末尾加 `patrol_lora_scale:=0.75` 替换为所选值。

## 3. “学会巡检”在车上具体指什么

- 方向统一以进入大棚时的第一帧位姿为准。先左转与在第一个十字路口左转有不同的触发条件。序号按沿途实际确认的地标计数。
- 巡一趟需要入行并回到实测行口。绕行闭环，或确认死胡同、沿实测轨迹退回，两种情况均计一趟。因感知不足或临时规划失败退回时不计数。“必须绕成闭环”仍要求车迹环绕作物。巡检可指定 1～20 趟。
- “这行再巡两趟”增加两趟，“这行一共巡三趟”会扣除本次记录中已完成的趟数。后续指令能引用最近实际进入的作物行。在车行驶时输入新任务会接管当前任务。重启、仿真时钟重置或定位跳变后需要重新确认行口。“左侧隔壁”会回到该行口，按入口方位寻找同向平行行。“隔壁”缺少左右方位或“多巡几圈”缺少具体次数时，节点停车提示补充，可以重新下达完整指令。
- 大棚前进速度上限可设 0.05～1.0 m/s，默认 0.16 m/s。小车在转弯、窄道、接近目标时会减速。1 m/s 是仿真允许的最高值，真车需另外测制动距离和控制延迟。
- 停车距离接受 0.05～50 m。观察到门或植株后才能确定其位置，路口需先勘测中心。4.5 m 起使用先前定位的目标和当前里程计来复核。近距复核容差为纵向 0.22 m、横向 0.25 m，这也限定了可分辨的停车精度。超出地图、停点被占用或不可达时会报告原因。当前地图中部分长距离无法实际停靠。
- 这批训练只有中文对话。作物品种能被解析为搜索目标，车上仍需视觉证据来确认。当前大棚已有草莓、生菜、盆栽。训练里出现黄瓜、番茄，并不会在场景中生成这些作物。

## 4. 数据怎么做

每一行 JSONL 是一道有答案的题。用户说一句话，标注者写出应有的动作顺序，每项“原文”逐字复制指令片段。速度、停车距离、圈数和“第一、第二”等信息保留在原文，运行时由程序确定单位和数值。

初版分为 320 条训练对话、32 条验证对话和 48 条留出测试对话。训练集包含 79 条有上一轮问答的对话。生成器组合任务、语序、说法、作物名和数值，三组使用不同的措辞。它们都来自同一个生成器，真实使用效果还要看没有参与设计的人怎么说。评测另有 10 条人工场景用例，包括过去跑过的指令。

自己补数据时，打开 `data\my_examples.jsonl`（首次使用新建 UTF-8 文本文件），每行写一个对象。例如下面一整行就是一条人工标注。

```json
{"指令":"用每秒八十厘米行驶，先向右拐，再到左门前六米停下","动作序列":[{"原文":"用每秒八十厘米行驶","动作":"调速"},{"原文":"先向右拐","动作":"转向","方位":"右"},{"原文":"再到左门前六米停下","动作":"靠近","目标":"门","方位":"左"}]}
```

优先收集自己实际说过、模型确实听错的原话，人工修正答案。一组例子只改一个关键点，例如“再巡两趟”和“一共巡两趟”、“靠近右门”和“从右门出去”。保留一份独立的新指令作验收。导航日志可以提供原话和错误线索，人工确认正确步骤后再用作标签。

## 5. 自己重新训练

在 Windows PowerShell 中执行。本机已经具备下述权重和依赖，复训通常从生成数据这一步开始。`$repo` 应指向包含 `src\camera_processing` 的 Windows 仓库。

```powershell
$root = 'D:\Qwen3-VL-2B\patrol_training'
$repo = 'C:\Users\pc\.codex\worktrees\66ef\robot_ws'
$py = 'E:\ANACONDA\envs\unet\python.exe'
& $py -B "$root\scripts\make_data.py" --root $root --repo $repo
& $py -u -B "$root\scripts\train.py" --root $root --epochs 4 --group 6
& $py -u -B "$root\scripts\evaluate.py" --root $root --repo $repo --split test
& $py -u -B "$root\scripts\evaluate.py" --root $root --repo $repo --split validation --calibrate
& $py -u -B "$root\scripts\evaluate.py" --root $root --repo $repo --split test --selected
```

第一条命令按固定种子重新生成三份数据，校验自己添加的原话，并将个人标注单独写入 `personal_train.jsonl`。第二条从官方权重开始，将答案的下一个 token 作为学习目标，只调整最后一层语言前馈网络中的两个低秩矩阵。RTX 4060 分批计算冻结网络的特征并缓存到 D 盘。显存紧张时可将 `--group 6` 改成 `--group 2`。每轮用验证集算损失，选损失最低的一轮。各轮都未超过原版验证成绩时，脚本会报告并保留原有适配器。

训练采用 rank 32、alpha 32、学习率 0.00045、4 条对话累计一次更新，训练参数共 262,144 个。训练结束后 `runs\` 保留该次训练，`adapter\` 放选中的版本。要恢复以前的适配器，关闭 Windows 推理服务，从对应运行目录复制 `patrol-lora.gguf` 到 `adapter\` 后重启。单纯使用已训练好的模型无需再训练。

第三条命令启动一个只监听 Windows 本机的临时语言服务，原版和微调版使用同一 Q4 权重、提示词、温度和解码格式。它用最后一条用户原话做单次解析，跟线上节点的第一轮模型调用一致，结束后自动关闭该服务。`exact_labels` 是动作、目标、引用文字等全都与标注一致的条数。`exact_plan` 要求编译后的动作顺序和每个参数全对。它衡量指令理解，完整导航另外在 Gazebo 验收。

后两条命令在 32 条验证对话上比较强度 0、0.25、0.5、0.75、1，优先看完整计划正确条数，平分时选完整标注命中更多的强度，再锁定强度跑测试集。`evaluation\scale.json` 记录选择依据。`evaluate.py --root $root --repo $repo --replay` 可通过项目的实际解析、校验及重试流程回放四条历史指令。

### 本次训练的成绩

采用第 4 轮。32 条验证对话的逐 token 损失从 0.6284 降到 0.2753。验证集上强度 0、0.5 和 0.75 的完整计划均为 14/32，0.75 的完整标注为 11/32，按上述规则选中。

| 验收项目 | 原版 | 训练版，强度 0.75 | 训练版，满强度 |
| --- | ---: | ---: | ---: |
| 48 条留出指令，完整执行计划正确 | 34/48 | **36/48** | 24/48 |
| 同一批指令，模型所有原始标注完全一致 | 2/48 | **7/48** | 8/48 |
| 10 条人工用例，完整执行计划正确 | 9/10 | **9/10** | 8/10 |
| 4 条历史日志原话，经过线上解析流程后的完整计划正确 | 3/4 | **3/4** | 未测 |

相对于原版，0.75 强度纠正了 5 条合成测试指令，也新增了 3 条错误。在人工用例里，“巡一下左边隔壁的作物行”出现原文引用错误。历史指令“以0.3米/秒的速度行驶，先左拐，遇到第一条作物行就进去巡一趟，走不通就退回来，再从右门出去”两种模型都给出了预期计划。历史指令中把“遇到第一条作物行，进去绕一圈”拆开的版本仍会产生多余步骤。后续补数据时，可以录入其他人自然说出的同类任务，并另外留出新的独立验收句。

本次另通过 94 项离线回归测试，包含占据地图逐步揭示、死胡同折返、再次巡检和日志同步。没有用真实小车测试制动，也没有对新版在 Ubuntu Gazebo 中统计整场任务成功率。

## 6. 换电脑或重建本机环境

训练脚本使用 CUDA 版 PyTorch（本机为 2.1.2）和支持 BF16 的 GPU。若附加库丢失，可用以下命令从 D 盘的离线包恢复。若官方训练权重丢失，下载脚本支持续传并校验完整文件的 SHA-256。推理使用原有的 Q4 GGUF、视觉投影器和 llama.cpp。

```powershell
& $py -m pip install --no-index --no-deps --find-links "$root\wheels" --target "$root\deps" safetensors==0.5.3 tokenizers==0.21.4 gguf==0.17.1
& $py -u -B "$root\scripts\fetch_base.py" --root $root
```

固定训练底座版本为 `89644892e4d85e24eaac8bacfd4f463576704203`。已有同版本配置但权重下载中断，可追加 `--source modelscope` 从 Qwen 魔搭仓库续传权重，仍使用相同的 SHA-256 校验。[官方模型](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) · [官方权重校验值](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/model.safetensors)
