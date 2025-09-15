function [labelStr, maxScore] = predTS(dataIn, lightSourceType) %#codegen
% this function is used to run prediction to determine Tissue vs non-tissue
% category
% dataIn - input reflectance spectrometry data with 2048 features
% lightSourceType - XenonABON, XenonABOFF, LEDABON, LEDABOFF
%

assert(all(size(dataIn) == [1 2048]));
assert(all(all(isa(dataIn, 'double'))), 'Data in the Excel sheet is not of type double');
%assert(isa(dataIn,'double'));
assert(ismember(lightSourceType, {'XenonABOFF', 'XenonABON', 'LEDABOFF', 'LEDABON'}));

% load in model
persistent Mdl
persistent MdlType

if isempty(MdlType) || ~strcmp(MdlType, lightSourceType)
    MdlType = lightSourceType;
    Mdl = load(strcat(lightSourceType, ".mat"));

    if strcmp(Mdl.result.mdl.ScoreTransform, 'none')
        Mdl.result.mdl = fitPosterior(Mdl.result.mdl);
    end
end

if isempty(Mdl)
    Mdl = load(strcat(lightSourceType, ".mat"));
    if strcmp(Mdl.result.mdl.ScoreTransform, 'none')
        Mdl.result.mdl = fitPosterior(Mdl.result.mdl);
    end
end

% run prediction
pipeopts = namedargs2cell( Mdl.result.pipesettings.Options );
[prediction, score] = basePipelinePredict( Mdl.result.mdl, dataIn, ...
    pipeopts{:});

labelStr  = cellstr(prediction);
maxScore  = max(score);
outputStr = sprintf('Label : %s \nScore : %f',labelStr{:},maxScore);

%Print label and score to stdout
fprintf('%s\n',outputStr);

end