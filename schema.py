target = 'CareerSatisfaction'

# TODO
feature_defaults = {'Hobby': '',
                    'OpenSource': '',
                    'Student': '',
                    'Employment': '',
                    'FormalEducation': '',
                    'CompanySize': '',
                    'YearsCoding': '',
                    'YearsCodingProf': '',
                    'JobSearchStatus': '',
                    'LastNewJob': '',
                    'WakeTime': '',
                    'HoursComputer': '',
                    'HoursOutside': '',
                    'SkipMeals': '',
                    'Exercise': '',
                    'Age': '',
                    'Dependents': ''}

feature_names = list(feature_defaults.keys())

record_defaults = [feature_defaults[name] for name in feature_names]
