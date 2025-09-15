function newPipelineTemplate( options )
    %newPipelineTemplate Generate a data preparation template for ml-framework
    %
    
    % Copyright 2021 The MathWorks Inc.

    arguments 
        options.ByType (1,1) string {mustBeMember(options.ByType,...
            ["Supervised", "Unsupervised" "RUL"])} = "Supervised"
        options.WriteToFile (1,1) logical = false;
        options.WriteDirectory (1,1) string = pwd
        options.FileName (1,1) string = "basePipeline"
    end
    
    %Checklist 
    supervisedCKList = string(['%%Common data preparation steps for supervised training\n'...
        '%%   - General Cleaning: Remove/impute instances with missing data,\n'...
        '%%     constant features, ...\n'...
        '%%   - Feature Engineering: Optionally transform, derive, buffer\n'...
        '%%     and aggregate features,...\n'...
        '%%   - Feature Scaling: If feature magnitudes differ, apply range\n' ...
        '%%     scaling, zscore normalization,...\n'...
        '%%   - Feature Selection: Apply data dependent feature selection methods\n\n\n']);
          
    %Reference 
    reference = string(['%%General Notes:\n'...
        '%%   - The toolbox provides convenience utilites for data preparation and ML.\n'...
        '%%     Please see ''util'' and ''baseml'' for additional details.\n' ...
        '%%   - See demos for examples of completed pipelines.\n'... 
        '%%   - The relative sequence of steps may vary. Consider creating separate\n'...
        '%%     pipelines for structural changes.\n'...
        '%%   - Ensure pipeline defines variables assigned in custom properties including\n'...
        '%%     a list of: featurenames and responsename (as strings), partition info (as logical),\n'...
        '%%     and pipeline settings (e.g. input arguments for provenance/reproduction ).\n\n\n']);
    

    %Partition 
    partitionInstructions = string(['%%Define Partition a holdout/test set( specify cross validation in later step )\n'...    
        '%%value = baseml.partition( value );\n\n']);
            
    %Response 
    responseInstructions = string(['%%Convert response/target to categorical (If Classification).\n'...
        '%%value.(responsename) = categorical( value.(responsename) )\n\n']);


    %Create 
    contents = struct();
    contents.signature          = "function value = " + options.FileName +"( data, options )\n\n";
    contents.arguments          = "arguments\ndata table\noptions.Parameter\nend\n\n";
    contents.bodyinstructions   = "%%Place your data preparation code here.\n";
    contents.assignment         = "value = data;\n\n";
    
    switch  options.ByType
        case "Supervised"
            contents.checklist          = supervisedCKList;
            contents.reference          = reference;
            contents.partition          = partitionInstructions; 
            contents.response           = responseInstructions;
            contents.instructions       = "%%Store Response, Features, Partition, and Pipeline options.\n";
            contents.customcreate       = string(['value = addprop(value, "Features", "table" );\n' ...
                'value = addprop(value, "Response", "table" );\n'...
                'value = addprop(value, "TrainingObservations", "table" );\n'...
                'value = addprop(value, "Options", "table" );\n\n']);
            contents.customassign       = string(['value.Properties.CustomProperties.Features = featurenames;\n' ...
                'value.Properties.CustomProperties.Response = responsename;\n' ...
                'value.Properties.CustomProperties.TrainingObservations = value.Partition == "Train";\n' ...
                'value.Properties.CustomProperties.Options = options;\n\n']);
        case "Unsupervised"
            contents.instructions       = "%%Store Features and Pipeline options. \n";
            contents.customcreate       = string(['value = addprop(value, "Features", "table" );\n' ...
                'value = addprop(value, "Options", "table" );\n\n']);
            contents.customassign       = string(['value.Properties.CustomProperties.Features = featurenames;\n' ...
                'value.Properties.CustomProperties.Options = options;\n\n']);
        case "RUL"
            contents.instructions       = "%%Store HealthIndicatorName, DataVariable, LifeTimeVariable, and Pipeline options.\n";
            contents.customcreate       = string(['value = addprop(value, "HealthIndicatorName", "table" );\n' ...
                'value = addprop(value, "DataVariable", "table" );\n'...
                'value = addprop(value, "LifeTimeVariable", "table" );\n' ...
                'value = addprop(value, "Options", "table" );\n\n']);
            contents.customassign       = string(['value.Properties.CustomProperties.HealthIndicatorName = healthIndicatorName;\n' ...
                'value.Properties.CustomProperties.DataVariable = dataVariable;\n' ...
                'value.Properties.CustomProperties.LifeTimeVariable = LifeTimeVariable;\n' ...
                'value.Properties.CustomProperties.Options = options;\n\n']);
    end
    
    contents.close  = "end %%function";
    
    %Concatenate 
    sections = string( fieldnames(contents) );

    template = "";
    for iSection = sections(:)'
        template = template + contents.(iSection);
    end %for iSection
 
    formattedtemplate = sprintf( template );
    
    if options.WriteToFile == true
        % Write pipeline file
        filename = options.FileName + ".m";
        pipelinefile = fullfile(options.WriteDirectory, filename);
        fid = fopen(pipelinefile, 'w');
        fidcloser = onCleanup(@()fclose(fid));
        fprintf(fid, '%s\n', char(formattedtemplate));
        
        edit(pipelinefile);
    else
        editorService = com.mathworks.mlservices.MLEditorServices; %#ok<JAPIMATHWORKS>
        editorApplication = editorService.getEditorApplication();
        editorApplication.newEditor( formattedtemplate );
    end
end %function

%{
Special Character
Single quotation mark ''
Percent character %%
Backslash \\
Alarm \a
Backspace \b
Form feed \f
New line \n
Carriage return \r
Horizontal tab \t
Vertical tab \v
Character whose Unicode® numeric value can be represented by the hexadecimal number, N \xN
Example: sprintf('\x5A') returns 'Z'
Character whose Unicode numeric value can be represented by the octal number, N \N
Example: sprintf('\132') returns 'Z'
%}
