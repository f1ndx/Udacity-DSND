$(document).ready( function() {

        $('#uploadFile').hide();
        $('#prediction-div').hide();

        $('#uploadFile').on('click', function()
        {
            var fd = new FormData();
            var files = $('#imgInp')[0].files[0];
            fd.append('file',files);
            
            $('#cover-spin').show();

            $.ajax({
                url: '/prediction',
                method: 'POST',
                data: fd,
                contentType: false,
                processData: false,
                success: function(response){
                    $('#prediction-div').show();
                    $('#prediction-text').text(response.predictionText);
                    $('#prediction-img').attr('src', response.predictionImage);
                },
                complete: function(response){
                    $('#cover-spin').hide();
                },
            });
        });

        $(document).on('change', '.btn-file :file', function() {
        var input = $(this),
            label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
        input.trigger('fileselect', [label]);
        });

        $('.btn-file :file').on('fileselect', function(event, label) {

            $('#prediction-div').hide();
            
            var input = $(this).parents('.input-group').find(':text'),
                log = label;
            
            if( input.length ) {
                input.val(log);
            } else {
                if( log ) alert(log);
            }
        
        });
        function readURL(input) {
            if (input.files && input.files[0]) {
                var reader = new FileReader();
                
                reader.onload = function (e) {
                    $('#img-upload').attr('src', e.target.result);
                    $('#uploadFile').show();
                }
                
                reader.readAsDataURL(input.files[0]);
            }
        }

        $("#imgInp").change(function(){
            readURL(this);
        });     
    });