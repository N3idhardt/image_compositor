import cv2
import numpy as np
import os
import json
import argparse
import glob
import re
import shutil

# --- 图像叠加核心函数 ---
def overlay_image_opencv(background, overlay, top_left_coordinates):
    """
    使用 OpenCV 和 NumPy 将一个小图叠加到背景图上的指定左上角位置。
    """
    if background is None or overlay is None:
        return background

    # 将背景图转换为 BGRA（如果需要）
    if background.shape[2] == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

    ov_height, ov_width = overlay.shape[:2]
    bg_height, bg_width = background.shape[:2]
    x1, y1 = top_left_coordinates
    
    # 确保位置合法，计算右下角坐标
    x2 = min(bg_width, x1 + ov_width)
    y2 = min(bg_height, y1 + ov_height)

    # 裁剪前景图和背景ROI，以确保尺寸匹配
    ov_cropped = overlay[0:y2-y1, 0:x2-x1]
    background_roi = background[y1:y2, x1:x2]

    # 如果裁剪后的区域是空的，跳过
    if ov_cropped.size == 0 or background_roi.size == 0:
        return background

    # 执行 Alpha 混合
    alpha_channel = ov_cropped[:, :, 3] / 255.0
    inverse_alpha = 1.0 - alpha_channel
    ov_colors = ov_cropped[:, :, :3]

    for c in range(0, 3):
        background_roi[:, :, c] = (background_roi[:, :, c].astype(float) * inverse_alpha + 
                                   ov_colors[:, :, c].astype(float) * alpha_channel).astype(np.uint8)
        
    background[y1:y2, x1:x2] = background_roi
    return background

# --- 辅助函数：从文件名中提取不带扩展名的文件名作为 Key ---
def get_filename_key(filepath):
    # os.path.splitext() 返回一个元组 ('filename', '.ext')
    # 我们只需要 'filename' 部分，即索引 0
    return os.path.splitext(os.path.basename(filepath))[0]

def process_directory(directory_path, output_base_dir, base_scan_dir):
    """
    处理指定目录。如果能合并则合并，否则复制所有原图。
    """
    directory_path = os.fspath(directory_path)
    offset_path = os.path.join(directory_path, "offset.json")
    if not os.path.exists(offset_path):
        return

    print(f"--- 正在处理目录: {directory_path} ---")

    try:
        with open(offset_path, 'r', encoding='utf-8-sig') as f:
            offsets_list = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误：解析 {offset_path} 失败，请检查 JSON 格式。错误信息: {e}")
        return
    
    offsets = {item['Key']: np.array(item['Value']) for item in offsets_list}

    # 查找所有需要的图片文件
    bg_files = sorted(glob.glob(os.path.join(directory_path, "0_*.png")))
    fg1_files = sorted(glob.glob(os.path.join(directory_path, "1_*.png")))
    fg2_files = sorted(glob.glob(os.path.join(directory_path, "2_*.png")))
    
    # 确定输出子目录路径
    relative_path = os.path.relpath(directory_path, start=base_scan_dir)
    output_dir = os.path.join(output_base_dir, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- 核心逻辑分支 ---
    if not bg_files:
        # 情况 A: 缺少背景图 (0_*)。将该目录下所有 PNG 原图复制到输出目录。
        print(f"警告：在 {directory_path} 中找不到任何以 0_ 开头的背景图。将复制所有 PNG 原图。")
        # 查找所有 PNG 文件
        all_png_files = glob.glob(os.path.join(directory_path, "*.png"))
        for file_path in all_png_files:
            dest_path = os.path.join(output_dir, os.path.basename(file_path))
            shutil.copy(file_path, dest_path) 
            print(f"  复制原图：{os.path.basename(file_path)}")
        return
    
    # 情况 B: 存在背景图，执行合并逻辑
    
    # 假设基准背景图是第一张 0_*.png
    background_path = bg_files[0] 
    base_bg_key = get_filename_key(background_path)
    
    if base_bg_key not in offsets:
         print(f"错误：背景图键 {base_bg_key} 在 JSON 中找不到对应的基础坐标，跳过此目录的合成。")
         return
    bg_base_offset = offsets[base_bg_key] 

    # --- 生成所有 0+1+2 组合 ---
    combinations_to_process = []
    
    if not fg1_files and not fg2_files:
        # 只有背景图，没有前景图（1_* 或 2_*），只复制背景图。
        print("信息：只有背景图，没有前景图（1_* 或 2_*），只复制背景图。")
        dest_path = os.path.join(output_dir, os.path.basename(directory_path)[2:]+"-"+os.path.basename(background_path))
        shutil.copy(background_path, dest_path)
        print(f"  复制背景原图：{os.path.basename(background_path)}")
        return
    elif not fg1_files:
        # 只有 0_* 和 2_*，组合 0+2
        for fg2_path in fg2_files:
             combinations_to_process.append((None, fg2_path))
    elif not fg2_files:
        # 只有 0_* 和 1_*，组合 0+1
        for fg1_path in fg1_files:
            combinations_to_process.append((fg1_path, None))
    else:
        # 生成所有 1_* 和所有 2_* 的笛卡尔积组合
        for fg1_path in fg1_files:
            for fg2_path in fg2_files:
                combinations_to_process.append((fg1_path, fg2_path))

    # 遍历所有组合并处理
    for fg1_path, fg2_path in combinations_to_process:
        
        output_filename = os.path.basename(directory_path)[2:]+"-"
        if fg1_path:
            output_filename += f"{get_filename_key(fg1_path)}-"
        if fg2_path:
            # 去掉最后的横杠，并添加 2 的文件名
            output_filename = output_filename.rstrip('-') 
            output_filename += f"-{get_filename_key(fg2_path)}"
            
        output_filename += ".png"
            
        output_path = os.path.join(output_dir, output_filename)

        # 1. 读取背景图（干净的开始）
        composite_img = cv2.imread(background_path, cv2.IMREAD_COLOR)

        # 2. 叠加 FG1 (如果存在)
        if fg1_path:
            fg1_key = get_filename_key(fg1_path)
            fg1_offset_raw = offsets.get(fg1_key)
            if fg1_offset_raw is not None:
                relative_coords_1 = fg1_offset_raw - bg_base_offset
                fg1_img = cv2.imread(fg1_path, cv2.IMREAD_UNCHANGED)
                if fg1_img is not None:
                    composite_img = overlay_image_opencv(composite_img, fg1_img, tuple(relative_coords_1))

        # 3. 叠加 FG2 (如果存在)
        if fg2_path:
            fg2_key = get_filename_key(fg2_path)
            fg2_offset_raw = offsets.get(fg2_key)
            if fg2_offset_raw is not None:
                relative_coords_2 = fg2_offset_raw - bg_base_offset
                fg2_img = cv2.imread(fg2_path, cv2.IMREAD_UNCHANGED)
                if fg2_img is not None:
                    composite_img = overlay_image_opencv(composite_img, fg2_img, tuple(relative_coords_2))

        # 保存最终结果
        try:
            cv2.imwrite(output_path, composite_img)
            print(f"  成功保存合成图片：{os.path.basename(output_path)}")
        except Exception as e:
            print(f"错误：无法写入图片 {output_path}。请检查输出目录权限。错误信息: {e}")


def main():
    parser = argparse.ArgumentParser(description="递归查找并合并指定目录下（包括子目录）的 PNG 图片。")
    parser.add_argument("directory1", type=str, help="第一个根目录路径。")
    parser.add_argument("directory2", type=str, nargs='?', default=None, help="第二个根目录路径（可选）。")
    
    args = parser.parse_args()

    directories_to_scan = [args.directory1]
    if args.directory2:
        directories_to_scan.append(args.directory2)

    output_base_dir = os.path.join(os.getcwd(), "Composite_Outputs")
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"\n所有合成图片将保存到主目录: {output_base_dir}\n")
    
    for base_dir in directories_to_scan:
        if not os.path.isdir(base_dir):
            print(f"错误：指定的路径不是一个有效的目录：{base_dir}")
            continue
            
        for root, dirs, files in os.walk(os.fspath(base_dir)):
            if "offset.json" in files:
                process_directory(root, output_base_dir, base_dir)

if __name__ == "__main__":
    main()
