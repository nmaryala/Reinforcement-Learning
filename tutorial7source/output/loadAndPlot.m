function loadAndPlot(fileName, ylab, yAxisPr)
    figure('Position', [100, 100, 300, 200]);
    colormap(jet);
    raw = load(fileName + ".csv");
    ms = raw(:,1);
    QSA = raw(:,2);
    QSA_stderr = raw(:,3);
    LS = raw(:,4);
    LS_stderr = raw(:,5);
    errorbar(ms, QSA, QSA_stderr, '-b', 'LineWidth', 2);
    hold on;
    errorbar(ms, LS, LS_stderr, ':r', 'LineWidth', 2);
    ylabel(ylab, 'interpreter','latex');
    xlabel("Amount of data, $m$", 'interpreter','latex');
    set(gca, 'XScale', 'log')
    xlim([min(ms), max(ms)]);
    if (yAxisPr == 1)
        ylim([0, 1]);
        legend("QSA", "LS");
    else
        ylim([0, 2.0]);
        plot([1, 100000], [1.25, 1.25], ':k');
        plot([1, 100000], [2.0, 2.0], ':k');
        legend("QSA", "LS", 'Location', 'SouthEast');
    end
    box off
    set(gcf,'color','w');
    export_fig(fileName + ".png", "-m3");
end