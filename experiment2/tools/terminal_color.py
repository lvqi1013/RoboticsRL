# terminal_color.py

class Colors:
    RESET = "\033[0m"

    # 普通颜色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # 高亮颜色
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"

    # 样式
    BOLD = "\033[1m"


def _print_color(prefix, msg, color):
    print(f"{color}{prefix}{msg}{Colors.RESET}")


# RL奖励
def reward(msg):
    """
    正奖励输出
    """
    _print_color(
        "[REWARD] ",
        msg,
        Colors.BRIGHT_GREEN
    )


# RL惩罚
def penalty(msg):
    """
    负奖励/惩罚输出
    """
    _print_color(
        "[PENALTY] ",
        msg,
        Colors.BRIGHT_RED
    )


# 错误
def error(msg):
    """
    错误信息
    """
    _print_color(
        "[ERROR] ",
        msg,
        Colors.RED + Colors.BOLD
    )


# 普通信息
def info(msg):
    """
    普通训练信息
    """
    _print_color(
        "[INFO] ",
        msg,
        Colors.BLUE
    )


# 警告
def warning(msg):
    """
    Warning
    """
    _print_color(
        "[WARNING] ",
        msg,
        Colors.YELLOW
    )


# Episode统计
def episode(ep, reward_value, steps):
    """
    RL episode结果
    """
    text = (
        f"Episode={ep}, "
        f"Reward={reward_value:.3f}, "
        f"Steps={steps}"
    )

    color = (
        Colors.BRIGHT_GREEN
        if reward_value >= 0
        else Colors.BRIGHT_RED
    )

    _print_color(
        "[EPISODE] ",
        text,
        color
    )


# Loss输出
def loss(value):
    """
    网络loss
    """
    _print_color(
        "[LOSS] ",
        f"{value:.6f}",
        Colors.MAGENTA
    )


# Metric输出
def metric(name, value):
    """
    指标
    """
    _print_color(
        "[METRIC] ",
        f"{name}: {value}",
        Colors.CYAN
    )