Y0range=[-0.5,0.5]
parameters=[-0.05,-1]
orders=[0.8,0.8]
TSim=0.025
numICs=8000
results = zeros(numICs*6, 2)
test_results = zeros(6000, 2)
for i = 1:numICs
    Y0 = Y0range(1) + (Y0range(2) -Y0range(1)) * rand(1, 2)
    [T, Y]=FOSpectrum(parameters, orders, TSim, Y0)
     results((i-1)*6+1:(i-1)*6+6, :) = Y(:, :);
end
csvwrite('FOSpectrum_train_0.8.csv', results);
for i = 1:1000
    Y0 = Y0range(1) + (Y0range(2) -Y0range(1)) * rand(1, 2)
    [T, Y]=FOSpectrum(parameters, orders, TSim, Y0)
     test_results((i-1)*6+1:(i-1)*6+6, :) = Y(:, :);
end
csvwrite('FOSpectrum_test_0.8.csv', test_results);