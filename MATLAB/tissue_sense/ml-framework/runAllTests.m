try
    % ---------------- FILL ME -------------------- 
    % RUN YOUR PROJECT SETUP HERE:
    % e.g. openProject("."); or install(); or addpath("blah"); etc.
%     disp("FILL ME");
    % ---------------- FILL ME --------------------
    openProject(".");
    cp = currentProject;
    rootFolder = cp.RootFolder;
    
    %% Importation
    import matlab.unittest.TestSuite
    import matlab.unittest.TestRunner
    import matlab.unittest.plugins.XMLPlugin
    import matlab.unittest.plugins.TAPPlugin
    import matlab.unittest.plugins.ToFile
    import matlab.unittest.plugins.CodeCoveragePlugin
    import matlab.unittest.plugins.codecoverage.CoberturaFormat

    % ---------------- FILL ME --------------------
    % e.g. suite = TestSuite.fromFolder(pwd,'IncludeSubFolders',true);
    % Or suite = TestSuite.fromPackage("mypackage"); etc.
%     disp("FILL ME");
    % ---------------- FILL ME --------------------
    
    suite = matlab.unittest.TestSuite.fromProject(cp);
    
    %% Add runner
    % For command windows output
    runner = TestRunner.withTextOutput();
    resultsDir = 'artifacts';
    
    %% Adding Junit Plugin
    % creating the XML file path
    resultsFile = fullfile(resultsDir,'JunitXMLResults.xml');
    % adding the plugin to the runner
    runner.addPlugin(XMLPlugin.producingJUnitFormat(resultsFile));
    
    %% Adding Coverage 
    % creating the coverage report path
    coverageFile1 = fullfile(resultsDir, 'coverageExperiments.xml');
    coverageFile2 = fullfile(resultsDir, 'coverageUtilities.xml');
    % creating the path to the functions to cover
%     src = fullfile('src');
    % adding the plugin to the runner
%     runner.addPlugin(CodeCoveragePlugin.forFolder(src,'IncludingSubfolders',true,...
%     'Producing', CoberturaFormat(coverageFile)));

    coverage1 = matlab.unittest.plugins.CodeCoveragePlugin.forFolder(rootFolder + "\ml-experiment", ...
        'IncludingSubfolders',true);
    runner.addPlugin(coverage1);

    coverage2 = matlab.unittest.plugins.CodeCoveragePlugin.forFolder(rootFolder + "\ml-utility", ...
        'IncludingSubfolders',true);
    runner.addPlugin(coverage2);
% 
%     coverage3 = matlab.unittest.plugins.CodeCoveragePlugin.forFolder(rootFolder + "\ml-utility", ...
%         'IncludingSubfolders',true);
%     runner.addPlugin(coverage3);

    %% run tests
    results = runner.run(suite);
    
    % Show a summary of results in case it's useful in the logs
    table(results)
catch e
    % Info if we errored
    disp(getReport(e,'extended'));
    if batchStartupOptionUsed
        exit(1);
    end
end

% This isn't really intended to be run from MATLAB, but if you do 
% batchStartupOptionUsed stops it exiting for you.
if batchStartupOptionUsed
    % Exit with 0 if test passed or 1 if test failed
    exit(any([results.Failed]))
end
