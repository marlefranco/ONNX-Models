function splitn = partitionDataBasedonFilenames(data, trainPercent)

% loop through every class
classes = unique(data.TargetType);

trainTF = false(height(data), 1);
testTF = false(height(data), 1);

for i = 1:length(classes)
    thisClassTF = data.TargetType==classes(i);
    thisClassTypes = unique(data.TargetType(thisClassTF) + data.TargetNumber(thisClassTF));
    numClassTypes = length(thisClassTypes);
    thisClassTrainIdx = randperm(numClassTypes, round(trainPercent/100*numClassTypes));
    thisClassTestIdx = setdiff(1:numClassTypes, thisClassTrainIdx);

    trainTF(ismember(data.TargetType + data.TargetNumber, ...
        thisClassTypes(thisClassTrainIdx))) = true;
    testTF(ismember(data.TargetType + data.TargetNumber, ...
        thisClassTypes(thisClassTestIdx))) = true;
end

splitn{1} = find(trainTF);
splitn{2} = find(testTF);

end