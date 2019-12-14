
# Code taken from
# https://thirld.com/blog/2015/05/30/julia-profiling-cheat-sheet/

using Profile
using ProfileView
using Serialization


function benchmark()
    # Any setup code goes here.
        function code_to_profile()
            let grad=grad, data=data, theta=theta, sim=sim, dtv=dtv
                simloglik_drill_data!(grad, data, theta, sim, dtv, true)
            end
        end

    # Run once, to force compilation.
    println("======================= First run:")
    @time code_to_profile()

    # Run a second time, with profiling.
    println("\n\n======================= Second run:")
    Profile.init(delay=0.01)
    Profile.clear()
    Profile.clear_malloc_data()
    @profile @time code_to_profile()

    # Write profile results to profile.bin.
    r = Profile.retrieve()
    f = open("profile.bin", "w")
    serialize(f, r)
    close(f)
end

benchmark()
f = open("profile.bin")
r = deserialize(f);
ProfileView.view(r[1], lidict=r[2])
