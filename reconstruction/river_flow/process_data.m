% Create a VideoReader object
video = VideoReader('../../data/Stavne_stor_17nov.mp4'); %

%% Get basic video properties
numFrames = video.NumFrames; % Total number of frames
frameRate = video.FrameRate; % Frames per second
width = video.Width; % Width of the frames
height = video.Height; % Height of the frames

%% Example: Reading and displaying the first frame

frame1 = read(video, 1); % Read the first frame
imshow(frame1); % Display the frame

%% Downsample video

% Create a VideoWriter object for the output video
outputVideo = VideoWriter('../../data/ilen_stor_17nov.mp4', 'MPEG-4');
outputVideo.FrameRate = video.FrameRate; % Keep the original frame rate
open(outputVideo);

% Initialize a 3D matrix to store downsampled grayscale frames
numFrames = floor(video.Duration * video.FrameRate); % Estimate total number of frames
downsampledFrames = zeros(video.Height / 2, video.Width / 2, numFrames, 'double'); % Preallocate memory for frames

%%

frameIdx = 1;

% Process each frame
while hasFrame(video)
    frame = readFrame(video); % Read the next frame
    
    % Convert the frame to grayscale
    grayFrame = rgb2gray(frame);
    
    % Downsample the frame by a factor of 2
    downsampledFrame = imresize(grayFrame, 0.5); % Scale factor of 0.5 for half resolution
    
    % Store the frame in the 3D matrix
    downsampledFrames(:, :, frameIdx) = downsampledFrame;
    
    % Write the modified frame to the new video file
    writeVideo(outputVideo, downsampledFrame);
    
    frameIdx = frameIdx + 1;
end

% Close the video writer object
close(outputVideo);

% Save the 3D matrix to a .mat file
save('../../data/ilen_stor_17nov.mat', 'downsampledFrames', '-v7.3');

% Display the first frame of the downsampled grayscale video to verify
imshow(downsampledFrames(:,:,1));