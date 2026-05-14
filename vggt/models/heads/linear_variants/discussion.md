Discussion A:




Model 12:12 AM
ThinkingThoughts
Expand to view model thoughts

chevron_right
Based on the images provided, your hypothesis is absolutely correct. The artifacts you are observing are a classic symptom of relying solely on the final layer tokens of a standard Vision Transformer (ViT) combined with a naive upsampling strategy like MLP + PixelShuffle.
Here is a breakdown of why this is happening, an evaluation of your proposed solution, and practical steps to fix it while maintaining your claim of a lightweight architecture.
1. Diagnosing the Artifacts
The artifacts in your images (especially visible in the second image) look like blocky, grid-like structures with sharp discontinuities between regions.
The ViT Bottleneck: A standard ViT (like ViT-B/16) downsamples the image into 
16
×
16
16×16
 non-overlapping patches right at the beginning. By the last layer, your tokens represent highly compressed, global semantic information at a 
1
/
16
1/16
 spatial resolution. High-frequency spatial details (exact edges, smooth gradients) have been lost.
The MLP + PixelShuffle Problem: An MLP applied to the token sequence acts essentially as a 
1
×
1
1×1
 convolution. It processes each 
1
/
16
1/16
 resolution token independently. When you apply a large PixelShuffle (e.g., 
16
×
16×
 upscaling) to map these channels to spatial pixels, it essentially expands one token into a 
16
×
16
16×16
 pixel block. Because the MLP did not allow adjacent patches to communicate, there is no spatial smoothing between these blocks, resulting in the visible "checkerboard" or "patch" borders you see in your results.
2. Evaluating Your Proposed Solution
Your idea: Take the output at layers 4, 7, 11, 23, concatenate them, and use an MLP + PixelShuffle.
Will it improve quantitative metrics (RMSE, etc.)? Yes, likely. Early layers contain low-level edge/texture features, while later layers contain high-level context. Combining them provides richer feature representation.
Will it fix the qualitative artifacts? No, it will not completely fix the blockiness. If you are using a standard ViT, layers 4, 7, 11, and 23 all have the exact same spatial resolution (e.g., 
1
/
16
1/16
). Concatenating them just creates a "deeper" feature map still at 
1
/
16
1/16
 resolution. If you pass this concatenated feature map through an MLP and do a massive 
16
×
16×
 PixelShuffle in one step, you will still generate independent 
16
×
16
16×16
 blocks with harsh boundaries.
3. How to Solve It (Keeping it Lightweight)
To match DPT's qualitative smoothness without DPT's heavy convolutional feature pyramids, you must introduce cross-patch spatial communication and handle the upsampling more gracefully. Here are the best ways to adapt your architecture:
Solution A: Progressive Upsampling with Lightweight Spatial Mixing (Highly Recommended)
Do not upsample 
16
×
16×
 in a single PixelShuffle step. Break it down and add a tiny bit of spatial mixing (convolutions) to smooth the boundaries. This remains much lighter than DPT.
Extract Intermediate Layers: Take features from layers (e.g., 4, 7, 11, 23).
Progressive Up-scaling: Instead of concatenating them all at 
1
/
16
1/16
, treat them as a pseudo-pyramid.
Take Layer 4, apply an MLP, and PixelShuffle 
2
×
2×
 (now at 
1
/
8
1/8
 resolution). Crucially: apply a simple 
3
×
3
3×3
 Depthwise Convolution here. The 
3
×
3
3×3
 conv allows neighboring pixels to blend, erasing the block boundaries.
Add or concatenate Layer 7 (upsampled to 
1
/
8
1/8
) to this.
Apply MLP -> PixelShuffle 
2
×
2×
 (now at 
1
/
4
1/4
 resolution) -> 
3
×
3
3×3
 Depthwise Conv.
Continue until full resolution.
Note: A 
3
×
3
3×3
 Depthwise Conv adds a negligible amount of parameters but completely destroys grid artifacts.
Solution B: The "SegFormer" Approach (All-MLP Decoder)
If you want to strictly claim an "MLP-based" decoder without standard convolutions, you can look at the SegFormer paper. However, this relies on having a hierarchical backbone (like Swin, Mix-Transformer, or PvT) that naturally outputs features at 
1
/
4
,
1
/
8
,
1
/
16
1/4,1/8,1/16
, and 
1
/
32
1/32
 resolutions.
If your backbone is already hierarchical:
Extract the 4 scales.
Use MLPs to project them all to the same channel dimension.
Use simple Bilinear Interpolation (not PixelShuffle) to upsample them all to the 
1
/
4
1/4
 resolution.
Concatenate them all together.
Apply a final MLP to fuse them and predict the depth map, followed by a final bilinear upsample to full resolution.
Interpolation naturally smooths patches better than a single large-factor PixelShuffle.
Solution C: If you MUST stick to Plain ViT + Single Step PixelShuffle
If you are locked into using a plain ViT and a single PixelShuffle step at the very end, you must smooth the output.
Concatenate layers 4, 7, 11, 23.
Apply your MLP.
Apply PixelShuffle to reach full resolution.
Post-processing Conv: Add two layers of 
3
×
3
3×3
 Convolutions (with ReLU) at the very end. This acts as an anti-aliasing / smoothing filter to blend the harsh block boundaries created by the PixelShuffle.
Summary of Action Plan:
Your core problem is that your upsampling mechanism maps 
1
/
16
1/16
 resolution tokens to full-res pixels without letting adjacent patches blend. Adding multi-scale features (layers 4, 7, 11) is a great idea for accuracy, but to fix the visual artifacts, you must either upsample progressively or insert a 
3
×
3
3×3
 convolution after your PixelShuffle to smooth the grid edges.
54.7s
info
Google AI models may make mistakes, so double-check outputs.
Use Arrow Up and Arrow Down to select a turn, Enter to jump to it, and Escape to return to the chat.
Start typing a prompt, use option + enter to append
google
Grounding with Google Search
1



Discussion B:



I am using the last layer tokens of ViT to estimate depth maps. Currently I can observe clear artifacts, as attached. Would you think this might be related to not using multi scale features?



Yes, you are absolutely on the right track. The artifacts visible in your depth map are a textbook example of what happens when relying solely on the last layer of a standard Vision Transformer (ViT) for dense prediction tasks like depth estimation.
Here is a breakdown of why this is happening and why multi-scale features are the standard solution:
1. The Root Cause: Patch Resolution and Loss of Spatial Detail
Standard ViTs (like ViT-B/16 or ViT-L/16) process images by dividing them into fixed-size, non-overlapping patches (typically 16x16 pixels).
When you extract only the last layer's tokens and reshape them into a 2D grid, you are working with a spatial resolution that is 1/16th of your original image resolution.
The last layer tokens are incredibly rich in semantic information (e.g., "this general area is a cup," "that area is a background plant"), but they have lost almost all fine-grained spatial resolution.
When you attempt to upsample this tiny 2D grid directly back to the original image size to create a depth map, the network doesn't know exactly where the sharp object boundaries lie within those 16x16 blocks. As a result, it essentially "guesses" depth at the patch level.
This is exactly why the edges of the central object and the objects on the left/right in your image look blocky, jagged, and resemble a staircase. You are literally seeing the original ViT patch grid manifesting in your output.
2. How Multi-Scale Features Fix This
To get crisp, pixel-accurate boundaries, you need to combine the high-level semantic understanding of the deeper layers with the high-resolution spatial details of the early layers.
In a standard convolutional neural network (CNN), this is usually done with a U-Net architecture. In the world of Vision Transformers, the breakthrough paper for this was DPT (Dense Prediction Transformers).
If you extract features from multiple stages of the ViT (for example, layers 3, 6, 9, and 12):
Early layers (e.g., Layer 3): The tokens still retain a lot of low-level information like edges, corners, and exact object boundaries, even though they don't quite know "what" the object is yet.
Late layers (e.g., Layer 12): The tokens know exactly what the objects are and their relative depths, but lack the sharp edges.
By using a feature pyramid or a decoder that fuses these multiple scales together, the network uses the early layers to "guide" the low-resolution depth map from the last layer, snapping the depth values cleanly to the actual object boundaries.
Recommendations for Your Architecture:
If you want to eliminate these artifacts, consider the following approaches:
Implement a Multi-Scale Decoder (like DPT): Instead of only taking the last layer, extract the token grids from 4 distinct depths in your ViT (e.g., 1/4, 2/4, 3/4, and the final layer). Use a set of convolutional layers (often called a Reassemble block) to project them to different resolutions (e.g., 1/4, 1/8, 1/16, 1/32 scales) and fuse them together using a standard feature pyramid network (FPN) decoder.
Hierarchical Transformers: If you have the flexibility to change your backbone, consider using a hierarchical transformer like Swin Transformer or PvT (Pyramid Vision Transformer). Unlike standard ViTs, these naturally produce multi-scale feature maps at different resolutions, making them much easier to hook up to standard dense-prediction decoders.
Cross-Attention Guidance: If you absolutely must stick to your current setup, you could look into using the high-resolution input image itself to guide the upsampling of your low-resolution tokens, perhaps via a guided filter or a learned cross-attention spatial upsampling module—though adding multi-scale tokens is generally a much more robust solution.


Yes, absolutely. The artifacts in this new set of depth maps are suffering from the exact same fundamental issue—a lack of multi-scale feature fusion—but they also reveal a second common issue related to how you are upsampling those features.
Here is a breakdown of the specific artifacts visible in these frames and what is causing them in your network:
1. The Primary Issue: Blocky/Jagged Edges (Lack of Multi-Scale)
Look closely at the edges of the vehicles (especially the roof racks, windshields, and tires) and the boundaries of the trees.
The Artifact: They look like they are made of Legos or Minecraft blocks. You can see distinct, hard, square-like boundaries instead of smooth, continuous lines following the objects.
The Cause: Just like in your first image, this confirms that your final depth map is being dictated entirely by a low-resolution feature grid (likely the 1/16th or 1/32nd resolution output of the final ViT layer). Because the network doesn't have access to the early layers (which contain the high-resolution pixel gradients and edge information), it cannot "snap" the depth estimates to the real physical contours of the trucks. It just upsamples a coarse grid, resulting in chunky, jagged blobs.
2. The Secondary Issue: Checkerboard Patterns (Upsampling Method)
Look at the background areas, particularly in the sky/distant areas (e.g., the lighter yellow/greenish patches in the top-right of the frames in the first and second rows).
The Artifact: You can see a very distinct, repeating grid or "checkerboard" texture of alternating light and dark squares superimposed over the smooth background.
The Cause: This is a classic artifact caused by the specific mathematical operation used to increase the resolution of your feature maps. This almost always happens when using Transposed Convolutions (sometimes called Deconvolution layers) with strides to upsample.
When the kernel size is not perfectly divisible by the stride, the network's overlapping filters cause an uneven distribution of "weight" on the output pixels, creating an artificial grid pattern.
Alternatively, if you are using a technique called "PixelShuffle" (depth-to-space) without proper convolution smoothing afterwards, it can also create this grid-like noise.
How to Fix This Architecture:
To get clean, smooth, and sharp depth maps from these frames, you need to address both the source of the spatial details and the method of upsampling:
Add Skip Connections (Multi-Scale Fusion): As discussed previously, you must route features from the early layers of your ViT (or CNN backbone) directly to your upsampling decoder. When the decoder tries to upsample the coarse truck features, it needs to concatenate or add the early-layer edge features to know exactly where the boundaries of the metal and glass are. (Look into the DPT - Dense Prediction Transformer architecture).
Change Your Upsampling Strategy (Fix the Checkerboard): If you are currently using ConvTranspose2d layers to scale up your images, replace them. The standard industry fix to eliminate checkerboard artifacts is called "Resize-Convolution".
Instead of a transposed convolution, first use standard Bilinear Interpolation (F.interpolate(..., mode='bilinear', align_corners=False)) to double the spatial resolution of your feature map.
Immediately follow that with a standard Conv2d layer (with stride=1 and padding=1) to process the upsampled features and smooth them out.
If you implement a multi-scale decoder using the resize-convolution approach, both the chunky Minecraft edges and the checkerboard sky artifacts will disappear.
User 11:50 PM
I am using ViT+ MLP + PixelShuffle. This is a key technical contribution of our paper, because we want to claim that DPT is too heavy, while using ViT+ MLP + PixelShuffle can achieve similar results (quantative and qualitive both) to DPT. While, now our qualitative results are not that good. What can I do to solve this problem? For example, taking the output at 4, 7, 11, 23, layers, concatanate them, and use a MLP + PixelShuffle to get the results?


This is a very strong research direction. Attempting to replace the heavy, convolutionally-dense decoder of DPT with a lightweight "All-MLP" or MLP + PixelShuffle decoder is highly relevant. In fact, this philosophy was successfully proven in semantic segmentation by SegFormer (which uses a lightweight All-MLP decoder).
However, if you simply concatenate layers 4, 7, 11, and 23 of a standard (isotropic) ViT and pass them through an MLP + PixelShuffle, you will likely still see these artifacts.
Here is why your proposed solution needs a slight tweak to work, and exactly how to adjust your architecture to fix the qualitative results while completely preserving your core contribution ("Our method is much lighter than DPT").
The Two Roadblocks in Your Current Plan
1. The Standard ViT Resolution Problem (Blocky Edges)
If you are using a standard ViT (like ViT-Base or ViT-Large), the tokens at layer 4, 7, 11, and 23 all have the exact same spatial resolution (1/16th of the original image).
Concatenating them gives the network better low-level vs. high-level feature understanding, but it does not give the network any higher spatial resolution. You are just making a thicker 1/16th feature map. When you push that thick feature map through a PixelShuffle, it will still generate blocky/Lego-like edges because it's upsampling from a coarse grid.
2. The PixelShuffle Checkerboard Problem
PixelShuffle (also known as Sub-Pixel Convolution) works by rearranging channel dimensions into spatial dimensions. Because it maps 
1
×
1
1×1
 spatial vectors directly into 
r
×
r
r×r
 pixel blocks (where 
r
r
 is your upscaling factor) without looking at neighboring tokens, it inherently creates the checkerboard artifacts visible in your sky/background.
How to Fix It (While Staying Lightweight)
To make your ViT + MLP + PixelShuffle approach achieve DPT-level qualitative results, you need to implement an All-MLP Multi-Scale Decoder (heavily inspired by SegFormer, but adapted for your setup) and add a "free" smoothing step.
Here is the exact pipeline I recommend for your paper:
Step 1: Unify the Dimensions (The MLP Step)
Extract your tokens from layers 4, 7, 11, and 23. Pass each layer's tokens through a simple Linear layer (MLP) to project them to the same channel dimension (e.g., 
C
=
256
C=256
). This is extremely lightweight.
Step 2: Create Artificial Scales (The Crucial Step)
Because your standard ViT only outputs at 1/16th scale, you must force these layers to act as multi-scale features before you fuse them.
Instead of heavy deconvolutional "Reassemble" blocks like DPT uses, do this:
Layer 4 (Low-level): Reshape to 2D grid. Upsample to 1/4th scale using F.interpolate(mode='bilinear'). (Bilinear is parameter-free and FLOP-cheap).
Layer 7: Reshape to 2D grid. Upsample to 1/8th scale using Bilinear.
Layer 11: Reshape to 2D grid, keep at 1/16th scale.
Layer 23 (High-level): Reshape to 2D grid, pool to 1/32nd scale (optional, or just keep at 1/16th).
Step 3: Fuse and PixelShuffle
Now, upsample all of those grids to a common resolution (e.g., 1/4th scale) using Bilinear interpolation.
Concatenate them all together.
Pass the concatenated block through an MLP (a 
1
×
1
1×1
 Conv) to fuse the features.
Use your PixelShuffle to upsample this fused 1/4th scale map to the final 1/1 resolution.
Step 4: The "Secret" to Fixing the Checkerboard
To fix the checkerboard artifact from the PixelShuffle without losing your claim of being lightweight, you need to mix the pixels locally. You cannot avoid this, but you can do it cheaply.
Do not use standard 
3
×
3
3×3
 Convolutions (that ruins your "lighter than DPT" claim).
Instead, use a 
3
×
3
3×3
 Depthwise Convolution (DWConv) immediately after the PixelShuffle.
A Depthwise Conv applies a single filter per channel. It has practically zero parameters and zero FLOPs compared to a regular convolution, but it perfectly smooths out the checkerboard artifacts caused by PixelShuffle by blending neighboring pixels.
How to frame this in your paper:
If you follow this architecture, your edges will become sharp (thanks to the Step 2 bilinear multi-scale creation) and the checkerboards will vanish (thanks to the Step 4 DWConv).
You can powerfully claim in your paper:
"Unlike DPT, which relies on computationally heavy ResNet-based 'RefineNet' feature fusion blocks, our decoder achieves multi-scale fusion using only lightweight Linear layers (MLPs) and parameter-free Bilinear interpolation. Furthermore, we replace heavy transposed convolutions with efficient PixelShuffle operations, smoothed by a near-zero-parameter Depthwise Convolution. This results in an architecture that matches DPT in qualitative sharpness and quantitative accuracy, but with a significantly lower computational budget."



But, even just holding BxHxWxC feature maps will be very expensive, is it True?



Yes, you are absolutely correct to point this out, and this is exactly the kind of critical thinking reviewers will look for in your paper.
Holding large 
B
×
H
×
W
×
C
B×H×W×C
 (or even 
B
×
H
4
×
W
4
×
C
B× 
4
H
​
 × 
4
W
​
 ×C
) feature maps in memory is expensive in terms of VRAM footprint and memory bandwidth.
However, you need to draw a very clear distinction in your paper between Memory Expense (VRAM) and Computational Expense (FLOPs/Parameters).
To do dense prediction (generating a high-resolution depth map), you cannot escape creating high-resolution spatial feature maps at some point in the network. The physics of the task demands it. The difference between your method and DPT is what you do with those maps once they are in memory.
Here is how you solve the memory expense problem to keep your architecture ultra-lightweight, and how you frame this advantage against DPT.
The Solution: The "Early Channel Reduction" Trick
If you extract features from layers 4, 7, 11, and 23 of a standard ViT, the channel dimension 
D
D
 is huge (e.g., 
D
=
768
D=768
 for ViT-Base, or 
D
=
1024
D=1024
 for ViT-Large).
If you upsample 
D
=
768
D=768
 channels to 
1
4
4
1
​
 
 resolution, your memory will explode. The trick is to radically shrink the channel dimension before you increase the spatial resolution.
Here is the exact step-by-step pipeline to keep the memory footprint tiny while achieving multi-scale sharpness:
1. The MLP Bottleneck (At 1/16th Scale)
Extract the four token grids (Layers 4, 7, 11, 23). They are currently at 
1
/
16
1/16
 scale, with 
D
=
768
D=768
 channels.
Immediately pass each through a lightweight Linear layer (MLP) to project the channels from 
768
→
C
l
i
g
h
t
768→C 
light
​
 
 (where 
C
l
i
g
h
t
C 
light
​
 
 is very small, e.g., 64 or 128).
Memory win: Operating MLPs at 
1
/
16
1/16
 scale is virtually free in both FLOPs and memory.
2. Cheap Upsampling to Common Resolution
Now you have four tiny feature maps of size 
.
Use parameter-free Bilinear Interpolation to resize them all to 
1
4
4
1
​
 
 scale. (Alternatively, you can use PixelShuffle here to expand the spatial dimension by consuming channels, but Bilinear is often smoother and mathematically cheaper for this intermediate step).
3. Concatenation
Concatenate the four 
1
4
4
1
​
 
 scale maps.
Because you reduced the channels first, your concatenated map is only 
 (since 
4
×
64
=
256
4×64=256
).
Memory win: Holding a 256-channel tensor at 
1
4
4
1
​
 
 resolution is standard, highly efficient, and takes a fraction of the memory of DPT's intermediate tensors.
4. The Final Fuse and PixelShuffle
Pass this concatenated tensor through one more MLP (
1
×
1
1×1
 Conv) to fuse the multi-scale features.
Apply your final PixelShuffle to upsample from 
1
4
4
1
​
 
 scale to the full 
1
/
1
1/1
 image resolution.
Add the lightweight Depthwise Convolution (DWConv) as discussed previously to smooth the checkerboard artifacts.
How to Defend This in Your Paper (vs. DPT)
By structuring your decoder this way, you perfectly counter the "isn't this expensive?" argument. You can claim the following massive advantages over DPT in your text:
DPT is computationally heavy: Once DPT creates its 
1
4
4
1
​
 
 and 
1
2
2
1
​
 
 resolution feature maps, it processes them using its "RefineNet" blocks, which consist of multiple standard 
3
×
3
3×3
 Convolutions. A standard 
3
×
3
3×3
 Conv at high resolution requires massive FLOPs because it multiplies across the spatial and channel dimensions simultaneously.
Your method is computationally cheap: You use MLPs (
1
×
1
1×1
 Convs). An MLP only mixes channels; it does not look at spatial neighbors. This reduces the FLOPs exponentially compared to DPT's 
3
×
3
3×3
 Convs.
Your spatial operations are essentially free: To handle the spatial mixing that DPT does with 
3
×
3
3×3
 Convs, you rely entirely on PixelShuffle (which is just a tensor reshape, 0 FLOPs) and Depthwise Convolution (which applies 1 tiny filter per channel, almost 0 FLOPs).
In Summary: You can state that while both DPT and your method must hold 
1
4
4
1
​
 
 resolution tensors in memory (a strict requirement for dense prediction), your MLP + PixelShuffle decoder operates on these tensors with a radically lower FLOP count and parameter count, utilizing early channel reduction to ensure the VRAM footprint remains strictly smaller than DPT.




Sounds great! While, do you think the last Depthwise Convolution is necessary?



It is an excellent question. When you are writing a paper whose core contribution is "our method is ultra-lightweight," you absolutely must justify the inclusion of every single layer.
To answer your question directly: Is it strictly necessary for the code to run? No. Is it necessary to fix the checkerboard artifacts and get DPT-level qualitative results? Yes, almost certainly.
Here is the technical reasoning you can use to justify it to yourself, and how to elegantly defend it in your paper against critical reviewers.
Why MLP + PixelShuffle Creates Checkerboards
To understand why the Depthwise Convolution (DWConv) is needed, we have to look at the math of what an MLP and a PixelShuffle actually do when paired together:
The MLP has no spatial awareness: An MLP (or 
1
×
1
1×1
 Conv) operates only along the channel dimension. It looks at a single pixel at a coarse resolution (say, 
1
4
4
1
​
 
 scale) and uses its channels to predict a vector of size 
r
2
r 
2
 
 (where 
r
r
 is your upsampling factor, e.g., 
r
=
4
r=4
).
PixelShuffle is just a reshape: It takes that 
1
×
1
×
r
2
1×1×r 
2
 
 vector and blindly unfolds it into an 
r
×
r
r×r
 spatial block (e.g., a 
4
×
4
4×4
 patch of depth pixels).
The Resulting Artifact: Because the MLP never looked at its neighboring coarse pixels, the 
4
×
4
4×4
 patch it generated has no mathematical obligation to seamlessly transition into the 
4
×
4
4×4
 patch generated right next to it.
This results in depth maps that look like mosaics or grids. This is exactly what is causing the faint grid-like texture in the background of your second set of images.
Why DWConv is the Perfect, "Free" Fix
You need some spatial operation to blend the edges of those 
r
×
r
r×r
 patches together so the depth map looks smooth and continuous.
If you use a standard 
3
×
3
3×3
 Convolution, your reviewers will rightly point out that computing a standard convolution at full 
1
/
1
1/1
 image resolution is computationally expensive, violating your paper's main claim.
This is why you use a 
3
×
3
3×3
 Depthwise Convolution.
A standard Conv multiplies across spatial dimensions and channel dimensions simultaneously.
A Depthwise Conv only multiplies across the spatial dimension (independently per channel).
Let's look at the absolute cost (The math you can put in your paper):
Let's assume your final PixelShuffle outputs the actual 1-channel Depth Map at full resolution (
H
×
W
×
1
H×W×1
).
If you apply a 
3
×
3
3×3
 DWConv to a 1-channel image, it requires exactly 9 parameters.
The FLOP cost is merely 
9
×
H
×
W
9×H×W
. In the context of modern neural networks, this is mathematically indistinguishable from zero.
Even if you output an intermediate feature map of 32 channels at full resolution before a final projection to 1 depth channel, the DWConv only takes 
32
×
9
=
288
32×9=288
 parameters.
How to frame this in your paper
You can turn the inclusion of this layer into a strength rather than a weakness. It proves you understand the mechanics of sub-pixel upsampling. You can write something like:
"While PixelShuffle provides a FLOP-free mechanism for spatial upsampling by reshaping channel dimensions, relying exclusively on 
1
×
1
1×1
 point-wise MLPs prior to upsampling deprives the network of local spatial continuity, leading to well-known sub-pixel grid artifacts (checkerboarding). To resolve this without breaking our lightweight computational budget, we introduce a single 
3
×
3
3×3
 Depthwise Convolution post-upsampling. Unlike the heavy 
3
×
3
3×3
 standard convolutions used in DPT's RefineNet, this depthwise operation enforces spatial smoothness with a negligible addition of parameters (e.g., < 300 params), fully preserving our model's efficiency advantage."
A Strategic Alternative (The "1/2 Scale" Trick)
If you are still nervous about doing any spatial operations at full 
1
/
1
1/1
 resolution, here is a widely accepted industry trick used in many depth and segmentation papers:
Don't predict at 1/1 resolution.
Use your MLP + PixelShuffle + DWConv to predict the depth map at 
1
/
2
1/2
 or 
1
/
4
1/4
 resolution.
As the absolute final step of your network, use parameter-free F.interpolate(mode='bilinear') to stretch that smooth, high-quality depth map to the final 
1
/
1
1/1
 image size.
Because depth maps are generally spatially smooth anyway (except at sharp object edges, which your multi-scale features already solved), bilinear interpolation from 
1
/
2
1/2
 scale works perfectly, looks great, and guarantees your FLOPs stay incredibly low.


