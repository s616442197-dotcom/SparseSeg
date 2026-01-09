// === Step 1: Canvas 变为 9 倍并居中 ===
// refine input
getDimensions(width, height, channels, slices, frames);
maxDim = maxOf(width, height);
scale = 1200 / maxDim;
scale = 1;
newWidth  = round(width  * scale);
newHeight = round(height * scale);
run("Size...", "width=" + newWidth + " height=" + newHeight + " average interpolation=Bilinear");

doCrop = getBoolean("是否执行 ROI Combine + Crop？");
if (doCrop) {
    getDimensions(width, height, channels, slices, frames);
	newWidth = width * 3;
	newHeight = height * 3;
	run("Canvas Size...", "width=" + newWidth + " height=" + newHeight + " position=Center");
	
	run("Linear Stack Alignment with SIFT", "initial_gaussian_blur=1.6 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Translation interpolate");
	
	//m = getNumber("请输入第一阶段 ROI 个数 m：", 3);
	waitForUser("请绘制 ROI 并点击 ROI Manager 的 Add，完成后点击 OK");
	
	roiManager("Select All");
	roiManager("Interpolate ROIs");
	
	n = roiManager("count");
	
	for (i = 0; i < n-1; i++) {
	    roiManager("select", i);
	//    setSlice(i+1);  // 切换到对应的第 i+1 张
	    run("Clear Outside", "slice");
	}
	roiManager("Select All");
	roiManager("Combine");
	getSelectionBounds(x, y, w, h);
	makeRectangle(x, y, w, h);
	run("Crop");
}


run("Duplicate...", "duplicate");
rename("stackA");
roiManager("Reset");


// === Step 6: 用户输入 n 并画 n 个 ROI ===
//n = getNumber("请输入第二阶段 ROI 个数 n：", 2);
waitForUser("请在 StackA 上绘制 ROI 并点击 ROI Manager 的 Add，完成后点击 OK");
roiManager("add & draw");
// === Step 7: 根据 StackA 大小创建全 0 Stack B 并填充 n 个 ROI ===
getDimensions(aW, aH, aC, aS, aF);
newImage("StackB", "8-bit black", aW, aH, aS);
for (i = 0; i < roiManager("count"); i++) {
	selectWindow("StackB");
    roiManager("Select", i);
    roiManager("Fill");
}

// === Step 6: 用户输入 n 并画 n 个 ROI ===
//n = getNumber("请输入第二阶段 ROI 个数 n：", 2);
roiManager("Reset");
waitForUser("请在 StackA 上绘制 negative ROI 并点击 ROI Manager 的 Add，完成后点击 OK");

// === Step 7: 根据 StackA 大小创建全 0 Stack B 并填充 n 个 ROI ===
getDimensions(aW, aH, aC, aS, aF);
newImage("StackC", "8-bit black", aW, aH, aS);
for (i = 0; i < roiManager("count"); i++) {
	selectWindow("StackC");
    roiManager("Select", i);
    roiManager("Fill");
}

// === Step 8: 保存 Stack A 和 Stack B ===
selectWindow("stackA");
saveAs("Tiff", getDirectory("Choose a Directory") + "StackA.tif");
selectWindow("StackB");
saveAs("Tiff", getDirectory("Choose a Directory") + "StackB.tif");
selectWindow("StackC");
saveAs("Tiff", getDirectory("Choose a Directory") + "StackC.tif");

showMessage("完成", "StackA 和 StackB 已保存！");