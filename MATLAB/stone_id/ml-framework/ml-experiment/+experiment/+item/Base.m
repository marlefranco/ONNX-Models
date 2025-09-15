classdef (Abstract) Base < matlab.mixin.SetGet & ...
        matlab.mixin.Heterogeneous
    %experiment.item.Base Abstract base class for experiment.item

    % Copyright 2021 The MathWorks Inc.
    
    properties
        pipeline
        pipelinesettings
        model
        modelsettings
    end
    
    properties
        Data table = table
    end %properties
        
    properties (Hidden)
        Prepared = false
    end
    
    properties (Dependent)
        features
        VariableNames
    end
    
    methods
        
        function obj = Base( varargin )
            %Base Construct an instance of this class
            
            if nargin > 0
                
                if ~isempty( varargin )
                    set(obj, varargin{:} )
                end
                
            end
            
        end %Base
        
    end %constructor
    
    
    methods
        
        function [result, info] = select( obj, options )
            %SELECT TODO DOC
            
            arguments
                obj
                options.Id (1,1) double {mustBeInteger}
                options.Model (1,1) double {mustBeInteger}
                options.Metadata (1,1) logical = false
            end
            
            if obj( options.Id ).Prepared == true && ~isempty(obj( options.Id ).model)
                
                if options.Metadata == true
                    
                    result = struct();
                    result.name = obj( options.Id ).describe.modelType( options.Model );
                    result.mdl = obj( options.Id ).model( options.Model ).mdl;
                    result.trialId = options.Id;
                    result.modelId = options.Model;
                   
                    %Return Custom Properties in Data Preparation Pipeline
                    pipesettings = obj( options.Id ).Data.Properties.CustomProperties;
                    fields = string( fieldnames(pipesettings) );
                    
                    result.pipesettings = struct();
                    if ~isempty( fields )
                        for iField = fields(:)'
                            result.pipesettings.( iField ) = pipesettings.(iField);
                        end %for iField 
                    end %if ~isempty( fieldnames )
                    
                    %Return property sweep
                    result.pipelinesweep = obj( options.Id ).pipelinesettings;
                    result.modelsweep = obj( options.Id ).modelsettings;
                    
                    %result.data = obj( options.Id ).result.Data;
                    info = obj( options.Id ).model( options.Model ).describe();
                    
                else
                    
                    result  = obj( options.Id ).model( options.Model ).mdl;
                    info    = obj( options.Id ).model( options.Model ).describe();
                    
                end %options.Metadata
                
            else
                
                result = [];
                info = [];
            end
            
        end %select
        
        function value = describe( obj )
            %DESCRIBE TODO DOC
            
            value = [];
            for i = 1:numel( obj )
                thisTrial = obj(i).describe();
                value = [value; thisTrial];  %#ok<AGROW>
            end
            
        end %describe
        
        function value = custom( obj, customProperties, options )
            %CUSTOM TODO DOC
            arguments
                obj
                customProperties (1,:) string
                options.Id (1,:) {mustBeInteger} = 1:numel(obj)
            end
            
            selection = options.Id(:)';
            
            values = cell( numel(selection),1 );
            for i = selection
                thisCustom = obj(i).Data.Properties.CustomProperties;
                
                propertiesInCustom = properties( thisCustom );
                
                value = table();
                for iCustom = customProperties
                    
                    tF = ismember( iCustom, propertiesInCustom );
                    
                    if tF == true
                        value.(iCustom) = thisCustom.(iCustom);
                    end
                    
                    values{ i } = value;
                    
                end
                
            end
            
            value = vertcat( values{:} );
            
        end %custom
        
        function value = sort( obj, options )
            %SORT TODO DOC
            
            arguments
                obj
                options.Metric (1,1) string {mustBeMember(options.Metric,["mseOnCV", "mseOnTest"])} = "mseOnCV"
            end
            
            results = obj.describe( );
            value   = sortrows( results, options.Metric );
            
        end %sort
        
    
    end %public
    
    methods
        function value = get.VariableNames( obj )
            value = string( obj.Data.Properties.VariableNames );
        end %function
        
        function value = get.features( obj )
            
            if ~isempty( obj.Data )
                
                if contains( "Features", properties(obj.Data.Properties.CustomProperties) )
                    value = obj.Data.Properties.CustomProperties.Features;
                else
                    warning('Custom properties not defined in data pipeline. \n%s', ...
                        'Setting default values for features and response');
                    if contains( "Partition", obj.Data.Properties.VariableNames )
                        value = string( obj.Data.Properties.VariableNames( 1:end-2 ) );
                        obj.Data = addprop(obj.Data, "Features", "table" );
                        
                        obj.Data.Properties.CustomProperties.Features = value;
                    else
                        value = string( obj.Data.Properties.VariableNames( 1:end-1 ) );
                        obj.Data = addprop(obj.Data, "Features", "table" );
                        
                        obj.Data.Properties.CustomProperties.Features = value;
                    end
                end
                
            else
                value = "";
            end
            
        end %function
        
    end %get
    
    
end %classdef





