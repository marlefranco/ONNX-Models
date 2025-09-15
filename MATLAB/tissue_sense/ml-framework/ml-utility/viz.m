classdef viz < baseml
    %VIZ Visualization methods for ML data preparation workflows.
    %   
    %   viz methods:
    %
    %   General
    %   scatterhistogram - scatterhistogram chart
    %   plotcorrelation  - correlation matrix chart
    %   plotmatrix       - scatter plot matrix
    %   boxchart         - box and whisker chart
    %   heatmap          - heatmap chart 
    %   scatter          - scatter plot
    %   bar              - grouped bar plot
    %
    %
    %   Regression
    %   predictactual    - prediction vs actual plot  
    %
    %
    %   Classification / Labeled 
    %   confusionchart   - confusion matrix chart
    %   parallelplot     - parallel coordinates chart
    %
    %
    %   Unsupervised
    %   silhouette       - cluster silhouette plot 
    %
    
    % MathWorks Consulting 2020
    
    methods ( Static )
        
        function silhouette( tbl, x, y )
            %SILHOUETTE Silhouette plot for clustered data
            %
            % Syntax:
            %   silhouette( tbl, features, label )
            %
            
            arguments
                tbl 
                x (1,:) string
                y (1,1) string 
            end
            
            %Data
            features = tbl(:, x).Variables;
            label    = tbl.(y);
            silhouette(features, label);
            
        end %function
        
        
        function h = scatterhistogram( tbl, x, y, custom, options )
            %SCATTERHISTOGRAM Scatterhistogram chart
            %
            % Syntax:
            %   ch = viz.scatterhistogram( tbl, x, y )
            %   ch = viz.scatterhistorgam( __, options...)
            %
            
            arguments
                tbl table
                x (1,1) string
                y (1,1) string
                custom.GroupVariable = ""
                options.?matlab.graphics.chart.ScatterHistogramChart
                options.HistogramDisplayStyle = "smooth"
                options.LineWidth = 1;
                
                options.LegendVisible = "on"
            end
            
            args    = namedargs2cell( options );
            
            if custom.GroupVariable ~= ""
                h = scatterhistogram( tbl, x, y,...
                    "GroupVariable", custom.GroupVariable, args{:} );
            else
                h = scatterhistogram( tbl, x, y, args{:} );  
            end
            
        end %scatterhistogram 
        
        
        function h = parallelplot( tbl, options )
            %PARALLELPLOT Parallel coordinates chart
            %
            % Syntax:
            %   ch = viz.parallelplot( tbl )
            %   ch = viz.parallelplot( __, options... )
            %
            
            arguments
                tbl table
                options.?matlab.graphics.chart.ParallelCoordinatesPlot
                options.Title = "Parallel Coordinates"
            end
            
            args    = namedargs2cell( options );
            h       = parallelplot( tbl, args{:} );
            
        end %function
        
        
        function  plotcorrelation( tbl, scatteropts, histopts, lineopts )
            %PLOTCORRELATION Correlation matrix chart
            %
            % Syntax:
            %   viz.plotcorrelation( tbl )
            %   viz.plotcorrelation( __, options... )
            %
            
            arguments
                tbl table
                scatteropts.Color   = [0 0.4470 0.7410]
                histopts.FaceColor  = [ 0.8500 0.3250 0.0980];
                histopts.EdgeColor  = "k";
                histopts.FaceAlpha  = 0.8;
                lineopts.LineColor  = [ 0.8500 0.3250 0.0980];
            end
            
            tF = varfun(@isnumeric, tbl, "OutputFormat", "uniform");
            tbl = tbl(:, tF);
            
            
            [~,~,h] = corrplot( tbl );

            scatterargs = namedargs2cell( scatteropts );
            histargs = namedargs2cell( histopts );
            vars = strrep( tbl.Properties.VariableNames, "_", " ");
            
            isHist = arrayfun(@(x)...
                isa(x,'matlab.graphics.chart.primitive.Histogram'),...
                h(:));
            
            %Lines 
            lines = findobj( [h.Parent], 'Tag', 'lsLines' );
            set(lines, 'Color', lineopts.LineColor)            
            set( h(~isHist), scatterargs{:});
            set( h(isHist), histargs{:});

           %Reset the labels 
           ax = findobj( gcf, 'Tag', 'PlotMatrixScatterAx' ); 
           ax=reshape(ax,sqrt(numel(ax)),sqrt(numel(ax)))';
           
            cols = ax( end, : );
            rows = ax( :, 1 );
            for iFeature = 1:numel(vars)  
               xlabel( cols( iFeature ), vars{iFeature} )
               ylabel( rows( iFeature ), vars{iFeature})
            end
            
        end %plotcorrelation
        
        
        function [s, h, ax] = plotmatrix( tbl, custom, scatteropts, histopts )
            %PLOTMATRIX Scatter plot matrix
            %
            % Syntax:
            %   [hs,hh,ax] = viz.plotmatrix( tbl )
            %   [hs,hh,ax] = viz.plotmatrix( __, options... )
            %
            
            arguments 
                tbl table
                custom.Title = ""
                custom.XLabel = ""
                custom.YLabel = ""
                custom.Skip = 1;
                scatteropts.?matlab.graphics.chart.primitive.Line
                scatteropts.Color   = [0 0.4470 0.7410];
                histopts.FaceColor  = [ 0.8500 0.3250 0.0980];
                histopts.FaceAlpha  = 0.8;
                
            end
            
            tF = varfun(@isnumeric, tbl, "OutputFormat", "uniform");
            tbl = tbl(:, tF);
             
            [s,axs, ax, h ] = plotmatrix( tbl.Variables );
            
            scatterargs = namedargs2cell( scatteropts );
            histargs = namedargs2cell( histopts );
            
            vars = strrep( tbl.Properties.VariableNames, "_", " ");
            
            cols = axs( end, : );
            rows = axs( :, 1 );
            for iFeature = 1:custom.Skip:numel(vars)  
               xlabel( cols( iFeature ), vars{iFeature} )
               ylabel( rows( iFeature ), vars{iFeature})
            end
     
            set(s, scatterargs{:});
            set(h, histargs{:});
            
            title(ax, custom.Title )
            xlabel(ax, custom.XLabel )
            
            ylabel(ax, custom.YLabel )
             
        end %function
        
        function h = correlationmatrix( tbl, options, custom ) 
           %CORRELATIONMATRIX Correlation matrix chart
           %
           % Syntax:
           %    ch = viz.correlationmatrix( tbl )
           %    ch = viz.correlationmatrix( __, options... )
           %
           
            arguments
               tbl table
               options.?matlab.graphics.chart.HeatmapChart
               options.Colormap = crameri( "vik" )
               options.ColorLimits = [-1 1];
               options.MissingDataColor = "w"
               options.GridVisible = "off"
               options.Title = "Correlation Matrix"
               custom.OmitNaNs = false;
            end
            
            tF = varfun(@isnumeric, tbl, "OutputF", "uni");
            
            if custom.OmitNaNs == true               
                cm = tril( corrcoef( tbl(:,tF).Variables, ...
                    "Rows", "complete"), 0 );       
            else
                cm = tril( corrcoef( tbl(:,tF).Variables ), 0 );
                
            end
 
            cm( cm == 0 ) = NaN;
            
            vars = strrep( tbl(:,tF).Properties.VariableNames, "_", " ");
            
            args = namedargs2cell( options );
            h = heatmap( vars, vars, cm, args{:} );
  
        end %correlationmatrix
        
        
        function [h,h2] = predictactual(tbl, actual, prediction, options, custom, ax)
            %PREDICTACTUAL Prediction vs actual plot  
            %
            % Syntax:
            %   [hs, hr] = viz.predictactual( tbl, actual, prediction )
            %   [hs, hr] = viz.predictactual( __, options...)
            %
            
            arguments
                tbl table
                actual (1,1) string
                prediction (1,1) string
                options.?matlab.graphics.chart.primitive.Line
                options.Marker = "."
                options.LineStyle = "none"
                options.MarkerSize = 25;
                custom.Grouping (1,1) string = ""
                custom.Axes (1,1) =  gobjects()
                custom.ColorOrder double = colororder();
                custom.Legend (1,1) logical = true
                ax.Title  = ""
                ax.XLabel = "Actual"
                ax.YLabel = "Predicted"
            end
            
            args    = namedargs2cell( options );
            
            
            if ~ishandle(custom.Axes)
                hax = gca();
            end
                
            if custom.Grouping ~= ""
               groups = string( unique( tbl.(custom.Grouping), 'stable' ) );
            else
               groups = "";
            end
                
            counter = 0;
            for iGroup = groups(:)'
                
                counter = counter + 1;
                
                if groups == ""
                    x = tbl.(actual);
                    y = tbl.(prediction);
                else
                    x = tbl.(actual)( tbl.(custom.Grouping) == iGroup );
                    y = tbl.(prediction)( tbl.(custom.Grouping) == iGroup );
                end
                
                h = [];
                if ishandle(custom.Axes)
                    h = [h line( x, y, args{:} )]; %#ok<AGROW>
                    h.Color = custom.ColorOrder( counter, : );
                    hold on
                else
                    h = [h line( custom.Axes, x, y, args{:}, "Parent", hax)]; %#ok<AGROW>
                    h.Color = custom.ColorOrder( counter, : );
                    hold on
                end
                
            end
            hold off
            
            colororder( h(1).Parent );
            reflineColor = custom.ColorOrder( 2,: );
            
            h2 = refline( 1, 0 );
            h2.Color = [.5 .5 .5];
            h2.LineWidth = 1;

            if custom.Legend == true
                
                if groups == ""
                    legend( ["obs"; "1:1"], "Location", "NorthWest" )
                else
                    legend( [groups; "1:1"], "Location", "NorthWest" )
                end
                
            end
            
            title( ax.Title )
            xlabel( ax.XLabel )
            ylabel( ax.YLabel )

        end %predictacutal
    
        
        function h = boxchart( tbl, x, y, options )
            %BOXCHART Box and whisker chart
            %
            % Syntax:
            %   ch = viz.boxchart( tbl, features )
            %   ch = viz.boxchart( tbl, features, grouping )
            %   ch = viz.boxchart( __, options...)
            %
            
            arguments 
               tbl 
               x (1,:) string 
               y (1,1) string = ""
               options.?matlab.graphics.chart.primitive.BoxChart
            end
            
            args = namedargs2cell( options );
            
            if y ~= ""
                
                if isdatetime(tbl.(y))
                    grouping = categorical( tbl.(y) );
                else
                    grouping = tbl(:,y).Variables;
                end
                
                h = boxchart( grouping, tbl(:,x).Variables, args{:} );
                title( x )
            else
                h = boxchart( tbl(:,x).Variables, args{:} );
                h.Parent.XTickLabel = x;
                h.Parent.XTickLabelRotation = 45;
            end
            
        end %boxchart
        
        
        function cm = confusionchart( tbl, trueLabel, predictedLabel, options )
            %CONFUSIONCHART Confusion matrix chart
            %
            % Syntax:
            %   ch = viz.confusionchart( tbl, actual, prediction )
            %   ch = viz.confusionchart( __, options...)
            %
            
            arguments
                tbl table
                trueLabel (1,1) string
                predictedLabel (1,1) string
                options.?mlearnlib.graphics.chart.ConfusionMatrixChart
                options.Title (1,1) string = ""
                options.Normalization (1,1) string {mustBeMember(options.Normalization,["absolute","column-normalized","row-normalized","total-normalized"])} = "absolute"
                options.RowSummary (1,1) string {mustBeMember(options.RowSummary,["off","absolute","row-normalized","total-normalized"])}  = "off"
                options.ColumnSummary (1,1) string {mustBeMember(options.ColumnSummary,["off","absolute","column-normalized","total-normalized"])} = "off"
            end
            
            args    = namedargs2cell( options );
            
            trueLabels = tbl.(trueLabel);
            predictedLabels = tbl.(predictedLabel);
            
            cm = confusionchart(trueLabels, predictedLabels, args{:} );
            
        end %confusionchart
        
        
        function h = heatmap( tbl, features, options )
            %HEATMAP Heatmap chart 
            %
            % Syntax:
            %   ch = viz.heatmap( tbl, features )
            %   ch = viz.heatmap( __, options...)
            %
            
            arguments
                tbl table
                features (1,:) string
                options.?matlab.graphics.chart.HeatmapChart
                options.Title (1,1) string = "Features"
            end
            
            h = heatmap( features, 1:height(tbl), ...
            tbl(:,features).Variables  );
            h.ColorScaling = 'scaledcolumns';
            h.GridVisible = 'off';
            h.YLabel = "Observation";
            h.XDisplayLabels = strrep( h.XDisplayLabels, "_"," " );
            h.YDisplayLabels = strings( size(h.YDisplayData) );
            title( options.Title + " ( ScaledColumns )" )
            
        end %function
        
        function scatter( tbl, features, options, custom )
            %SCATTER Scatter plot
            %
            % Syntax:
            %   viz.scatter( tbl )
            %   viz.scatter( __, options...)
            %
            
            arguments
                tbl table
                features (1,2) string = tbl.Properties.VariableNames(1:2)
                options.?matlab.graphics.chart.primitive.Scatter
                options.MarkerFaceAlpha = .75
                options.MarkerEdgeAlpha =.8
                custom.Label (1,1) string = ""
                custom.Title (1,1) = ""
                custom.XLabel (1,1) string = features(1)
                custom.YLabel (1,1) string = features(2)
                custom.Colormap = lines
                custom.Text (1,1) = ""
            end
            
            args  = namedargs2cell( options );
            
            if custom.Label ~= ""
                scatter(tbl.(features(1)), tbl.(features(2)), 50, tbl.(custom.Label), 'filled', ...
                    args{:} );
            else
                scatter(tbl.(features(1)), tbl.(features(2)), 50, 'filled', ...
                    args{:} );
            end
            
            if custom.Text ~= ""
                text( tbl.(features(1)),tbl.(features(2)), ...
                   tbl.(custom.Text), "FontSize", 8 )
            end
            
            colormap( custom.Colormap )
            xlabel( custom.XLabel )
            ylabel( custom.YLabel )
            title( custom.Title )
            
        end %function 
        
        
        function bar( tbl, features, label,options, custom )
            %BAR Grouped bar plot
            %
            % Syntax:
            %   viz.bar( tbl, features, labels )   
            %   viz.bar( __, options...)   
            %
            
            arguments
                tbl table
                features (1,1) string 
                label (1,1) string 
                options.?matlab.graphics.chart.primitive.Bar
                options.FaceColor = 'flat'
                options.FaceAlpha = .75
                custom.Statistic (1,1) string = "mean"
                custom.Colormap = [];
                custom.Title (1,1) = ""
                custom.XLabel (1,1) string = "Group"
                custom.YLabel (1,1) string = features
            end
            
           args  = namedargs2cell( options );
            
            value = groupsummary(tbl, label, custom.Statistic, features);
                        
            vars = string(value.Properties.VariableNames);
            features = vars( contains( vars, features ) );
            
            bc = bar( value{:,features}, args{:} );
            
            if isempty(custom.Colormap)
                colors = lines( numel(unique( tbl.(label) )) );
            else
                colors = custom.Colormap;
            end
    
            for iColor = 1:size(colors,1)
                bc.CData(iColor,:) = colors(iColor,:);
            end
            
            xlabel( custom.XLabel )
            ylabel( strrep(label,"_"," ") )
            title( custom.Title )
            
        end %function 
        
    end %public 
end %classdef

