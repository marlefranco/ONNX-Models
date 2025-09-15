function prj = createProjectFromTemplateDev(options)
% CREATEPROJECTFROMTEMPLATE Create a project from a default framework template
%
%   CREATEPROJECTFROMTEMPLATE creates a project in the current working
%   directory with the same name as the current folder
%
%   CREATEPROJECTFROMTEMPLATE('Name', value) accepts name-value pairs
%
%   Name-Value Pairs:
%   - "Name", name of project (optional, 1x1 string)
%
%   Examples: - createProjectFromTemplate() -
%   createProjectFromTemplate("Name", "Hello_Project")
%   createProjectFromTemplate("Name", "Hello_Project", ...
%       "IncludePipelineTemplates", false)

% Copyright 2021 The MathWorks Inc.

    arguments
        options.Name(1, 1) string = getDefaultProjectName
        options.IncludePipelineTemplates (1,1) logical = true
        options.InitializeGit(1,1) logical = true;
    end

    % Create a git repo (if none exists), project with the right format, and
    % add the various files to it.
    if options.InitializeGit
        [~, ~] = system("git init");
        mainprj = currentProject;
        prj = matlab.project.createProject(pwd);
        prj.Name = options.Name;
        
        [~, ~] = system("git mv Blank_project.prj " + prj.Name+".prj");
        [~, ~] = system("git submodule add https://insidelabs-git.mathworks.com/nhowes/ml-framework.git");  
    else
        mainprj = currentProject;
        prj = matlab.project.createProject(pwd);
        prj.Name = options.Name;
        
        movefile( string(dir('*.prj').name), options.Name + ".prj" );
    end

    % Add framework as reference project
    addReference(prj, mainprj, "relative");
    
    folderlist = ["Data" "Tests" "Demos" "Pipelines" "Helper" "Prototype" "Results"];
    pipelines = ["Training", "Prediction"];
    for folder = folderlist(:)'
        mkdir(folder);
        
        addFile(prj, folder);
        addPath(prj, folder);
        
        if folder == "Pipelines"
            for pipeline = pipelines(:)'
                pipePath = fullfile(prj.RootFolder,folder,pipeline);
                mkdir(pipePath);
                addFile(prj, pipePath);
                addPath(prj, pipePath);
                
                if options.IncludePipelineTemplates
                    pipename = "basePipeline" + extractBefore(pipeline, "ing"|"ion");
                    newPipelineTemplate("ByType", "Supervised", ...
                        "WriteToFile", true, ...
                        "WriteDirectory", pipePath, ...
                        "FileName", pipename);
                    addFile(prj, fullfile(pipePath, pipename+".m"));
                end
            end
        end
    end        
    
    % Make sure using fixed path manager
    close(prj);    
    matlab.project.convertDefinitionFiles(pwd, matlab.project.DefinitionFiles.FixedPathMultiFile);
    prj = openProject(pwd);
    
    % Be polite
    if ~nargout 
        clearvars
    end

end
function defaultprojectname = getDefaultProjectName
% Name of current folder in current folder
    [~, prjname, ~] = fileparts(pwd);
    defaultprojectname = fullfile(pwd, prjname);
    
end