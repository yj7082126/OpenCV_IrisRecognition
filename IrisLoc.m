
function c = IrisLoc(url, mode)
origin_img = imread(url);
bm=edge(origin_img,'canny');
c = {}

%% Find inner boundary (pupil border)
[pupil_center, pupil_radii] = imfindcircles(bm,[20 70], ...
    'ObjectPolarity','dark','Method','TwoStage','Sensitivity',0.81,'EdgeThreshold',0.1);


% Find training outer boundary (iris)
if mode == 1
	[iris_center, iris_radii] = imfindcircles(bm,[90 140], ...
		'ObjectPolarity','dark','Method','TwoStage','Sensitivity',0.96,'EdgeThreshold',0.1);
else
	[iris_center, iris_radii] = imfindcircles(bm,[90 170], ...
    'ObjectPolarity','dark','Method','TwoStage','Sensitivity',0.99,'EdgeThreshold',0.1);
end

c = [c, pupil_center];
c = [c, pupil_radii];
c = [c, iris_center];
c = [c, iris_radii];

end
%     imshow(origin_img);
%     viscircles(pupil_center,pupil_radii);
%     viscircles(iris_center,iris_radii);

%save_name = fullfile(current_path, strcat('MatLab_circle_', goal), strcat(goal, '_circle_', num2str(counter), '.mat'));
%save(save_name, 'c');
%counter = counter+1;
