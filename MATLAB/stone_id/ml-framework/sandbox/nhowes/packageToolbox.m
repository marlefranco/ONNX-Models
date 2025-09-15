
%% Setup

prj     = matlab.project.rootProject;
rootdir = prj.RootFolder;
docdir  = fullfile(rootdir, "doc" );

%% Classes for documentation 

%m-code classdef files
classesForDocumentation = [...
    "experiment.Regression"
    "experiment.Classification"
    "fitr"
    "fitc"
    ];

%mlx scripts
scriptsForDocumentation = [...
    "MainPage.mlx"
    "GettingStarted.mlx"
    "MLExperimentClassList.mlx"
    "MLUtilityClassList.mlx"
    ];


%% Generate html doc from live editor scripts

for iScript = scriptsForDocumentation(:)'
    
    html = strrep(iScript,'.mlx','.html');
    
    scriptlocation = fullfile(docdir, iScript );
    outputlocation = fullfile(docdir, html );
    
    matlab.internal.liveeditor.openAndConvert( char(scriptlocation), ...
        char(outputlocation));
end
    
%% Generate html on class documentation 

for iClass = classesForDocumentation(:)'
    
    thisFile = fullfile(docdir, strcat(iClass,".html") );

    html = help2html( iClass, '-doc' );
    html = replaceBetween(html,"href=""", ".css"">", "helpwin");
    html = replaceBetween(html,"<tr class=""subheader"">", "/tr>","", "Boundaries","inclusive");
    
    % Write the HTML file
    fid = fopen(thisFile,'w');
    fprintf(fid,'%s',html);
    fclose(fid);

end

%% Search database 

builddocsearchdb( docdir )
