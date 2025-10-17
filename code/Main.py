from train_eval_new import train, eval, select_best
def main(model_config=None):
    # 基础模型配置，包括DBS3TAN中用到的额外参数
    modelConfig = {
        "state": "train",  # 训练、评估或选择最佳模型
        "epoch": 20,
        "band": 188,  # 可根据需要调整不同频带
        "multiplier": 2,
        "seed": 1,
        "batch_size": 64,
        "group_length": 20,
        "depth": 4,
        "heads": 4,
        "dim_head": 64,
        "mlp_dim": 64,
        "adjust": False,
        "channel": 128,
        "lr": 1e-4,
        "epsilon": 5,
        "grad_clip": 1,
        "device": "cpu",  # 设置为CPU
        #"in_cha": 193,      # 定义Net模型的输入通道数
        "patch": 7,       # ViT中的补丁大小
        "m": 5,
        "state_size": 16,
        "layer": 1,
        "delta": 0.1,
        "num_class": 2,  # 输出的类别数量，用于线性层
        "training_load_weight": None,
        "save_dir": "./Checkpoint/",
        "test_load_weight": "ckpt_0_.pt",
        "path": "beach1.mat"
    }

    if model_config is not None:
        modelConfig.update(model_config)  # 更新传入的配置

    # 动态调整group_length
    if modelConfig["group_length"] is None:
        modelConfig["group_length"] = 20

    # 根据模型状态执行相应操作
    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["state"] == "eval":
        eval(modelConfig)
    elif modelConfig["state"] == "select_best":
        select_best(modelConfig)
    else:
        eval(modelConfig)
if __name__ == '__main__':
    main()
