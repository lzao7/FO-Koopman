Y0range=[0.1,0.5]
parameters=[35,3,28,-7]
orders=[0.9,0.9,0.9]
TSim=0.025
numICs=15000
results = zeros(numICs*6, 3)
test_results = zeros(6000, 3)
for i = 1:numICs
    Y0 = Y0range(1) + (Y0range(2) -Y0range(1)) * rand(1, 3)
    [T, Y]=FOChen(parameters, orders, TSim, Y0)
     results((i-1)*6+1:(i-1)*6+6, :) = Y(:, :);
end
csvwrite('FOChen_train_0.90.csv', results);
for i = 1:1000
    Y0 = Y0range(1) + (Y0range(2) -Y0range(1)) * rand(1, 3)
    [T, Y]=FOChen(parameters, orders, TSim, Y0)
     test_results((i-1)*6+1:(i-1)*6+6, :) = Y(:, :);
end
csvwrite('FOChen_test_0.90.csv', test_results);