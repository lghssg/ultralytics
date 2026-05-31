"""
YOLO 姿态检测脚本 - 检测视频中的人物姿势并绘制骨架
"""
import os
import glob
from pathlib import Path
from ultralytics import YOLO


def process_video(video_path: str, model: YOLO) -> str:
    """
    处理单个视频，检测人物姿势并绘制骨架

    Args:
        video_path: 视频文件路径
        model: YOLO 模型实例

    Returns:
        输出视频路径
    """
    video_path = Path(video_path)
    output_name = f"{video_path.stem}_new.mp4"

    print(f"正在处理: {video_path.name}")
    print(f"输出文件: {output_name}")

    # 运行姿态检测
    results = model.predict(
        source=str(video_path),
        save=True,
        project=".",
        name="temp_output",
        exist_ok=True,
        verbose=False
    )

    # YOLO 默认输出为 avi 格式，查找输出文件
    temp_dir = Path("temp_output")
    if temp_dir.exists():
        # 查找所有视频文件
        output_files = list(temp_dir.glob("*.avi")) + list(temp_dir.glob("*.mp4"))
        if output_files:
            temp_output = output_files[0]
            # 重命名为目标文件
            os.rename(str(temp_output), output_name)
            print(f"处理完成: {output_name}")
        else:
            print("警告: 未找到输出视频文件")

        # 清理临时目录
        import shutil
        shutil.rmtree("temp_output", ignore_errors=True)
    else:
        print("警告: 临时输出目录不存在")

    return output_name


def main():
    """主函数 - 处理当前目录下所有视频文件"""
    current_dir = Path(".")

    # 支持的视频格式
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    # 获取所有视频文件（排除已处理的 _new.mp4 文件）
    video_files = [f for f in current_dir.iterdir()
                   if f.is_file()
                   and f.suffix.lower() in video_extensions
                   and not f.stem.endswith("_new")]

    if not video_files:
        print("当前目录下没有找到视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件")

    # 加载 YOLO 姿态检测模型
    print("正在加载 YOLO 姿态检测模型...")
    model = YOLO("yolo11n-pose.pt")  # 使用 nano 版本的姿态检测模型

    # 处理每个视频
    for video_file in video_files:
        try:
            process_video(video_file, model)
        except Exception as e:
            print(f"处理 {video_file.name} 时出错: {e}")

    print("\n所有视频处理完成!")


if __name__ == "__main__":
    main()
