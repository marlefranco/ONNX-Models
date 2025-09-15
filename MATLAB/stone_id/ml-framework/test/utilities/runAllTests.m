function runAllTests()
%% Test Runner and Test Suite
%Build test runner and test suite. These allow you to customize how the suite is run and which tests to run.

%Copyright 2021 The MathWorks Inc.

%% Import Packages
import matlab.unittest.TestRunner
import matlab.unittest.plugins.XMLPlugin
import matlab.unittest.plugins.TAPPlugin
import matlab.unittest.plugins.TestReportPlugin
import matlab.unittest.plugins.CodeCoveragePlugin
import matlab.unittest.plugins.codecoverage.CoberturaFormat
import matlab.unittest.TestSuite

%% Add folders to path

% Find project path based on utilties folder for CI
if batchStartupOptionUsed
    projectPath = fileparts(fileparts(pwd));
    openProject(projectPath);
end
% openProject(".")
cp = currentProject;

%% Run Tests
try
    
    %% Create Test Runner
    % For command windows output
    runner = TestRunner.withTextOutput();
    resultsDir = 'artifacts';
    
    %% Create reports only if used in CI
    if batchStartupOptionUsed
        % Generate artifacts folder
        %% Adding Junit Plugin
        % creating the XML file path
        resultsFile = fullfile(resultsDir,'JunitXMLResults.xml');
        % adding the plugin to the runner
        runner.addPlugin(XMLPlugin.producingJUnitFormat(resultsFile));
        
        %% Adding Coverage
        % creating the coverage report path
        coverageExperiments = fullfile(resultsDir, 'coverageExperiments.xml');
        coverageUtilities = fullfile(resultsDir, 'coverageUtilities.xml');
        
        % creating the path to the functions to cover
        coverage1 = CodeCoveragePlugin.forPackage("experiment", ...
            "IncludingSubpackages", true, ...
            "Producing", CoberturaFormat(coverageExperiments));
        runner.addPlugin(coverage1);
        
        coverage2 = CodeCoveragePlugin.forFolder("ml-utility", ...
            "IncludingSubfolders", true, ...
            "Producing", CoberturaFormat(coverageUtilities));
        runner.addPlugin(coverage2);
        
    end
    
    %% Create the test suite from the test packages
    suite = TestSuite.fromProject(cp);
    
    %% Run Tests
    %Run the tests and see results
    results = run(runner, suite);
    
    %% Display results
    disp(results.table);
    
    %% Did results all pass?
    if all([results.Passed])
        disp("All Tests Passed");
    else
        warning("mlexperiments:runTestSuite","Not all tests passed. Check results.");
    end
catch e
    % Info if we errored
    disp(getReport(e,'extended'));
    if batchStartupOptionUsed
        exit(1);
    end
end %try/catch

% This isn't really intended to be run from MATLAB, but if you do
% batchStartupOptionUsed stops it exiting for you.
if batchStartupOptionUsed
    % Exit with 0 if test passed or 1 if test failed
    exit(any([results.Failed]))
end
end %function
