% Store cumulative days passed before the 1st of each month in a leap year (2024)
% Format: cum_days(Month, DaysPassed)
cum_days(1, 0).    % January
cum_days(2, 31).   % February (31 days in Jan)
cum_days(3, 60).   % March (31 Jan + 29 Feb)
cum_days(4, 91).   % April (60 + 31 Mar)
cum_days(5, 121).  % May (91 + 30 Apr)
cum_days(6, 152).  % June (121 + 31 May)
cum_days(7, 182).  % July (152 + 30 Jun)
cum_days(8, 213).  % August (182 + 31 Jul)
cum_days(9, 244).  % September (213 + 31 Aug)
cum_days(10, 274). % October (244 + 30 Sep)
cum_days(11, 305). % November (274 + 31 Oct)
cum_days(12, 335). % December (305 + 30 Nov)

% Helper rule: Convert a date string "DDMM" to the absolute day of the year (1 to 366)
date_to_days(DateStr, TotalDays) :-
    % Extract the first 2 characters for the Day and next 2 for the Month
    sub_string(DateStr, 0, 2, _, DayStr),
    sub_string(DateStr, 2, 2, _, MonthStr),
    
    % Convert the extracted string variables into numbers
    number_string(Day, DayStr),
    number_string(Month, MonthStr),
    
    % Look up the cumulative days for that specific month
    cum_days(Month, PrevDays),
    
    % Add the day of the current month to get the absolute day of the year
    TotalDays is PrevDays + Day.

% Main rule: Calculate the interval between two dates and output the result
interval(Date1, Date2) :-
    % Convert both input dates to absolute days of the year using our helper rule
    date_to_days(Date1, Days1),
    date_to_days(Date2, Days2),
    
    % Calculate the absolute difference between them
    Diff is abs(Days2 - Days1),
    
    % Print the result
    writeln(Diff).